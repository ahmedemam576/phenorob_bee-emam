import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import json
from collections import defaultdict

from rfdetr import RFDETRBase
import norfair
from norfair import Detection, Tracker

class BeeTracker:
    def __init__(self, model_path, conf_threshold=0.5, distance_threshold=30, hit_counter_max=10, initialization_delay=3, frame_rate=30):
        """
        Initialize the bee tracking system with Norfair tracker
        
        Args:
            model_path: Path to RF-DETR model weights
            conf_threshold: Confidence threshold for detections
            distance_threshold: Maximum distance for matching detections to tracks (pixels)
            hit_counter_max: Maximum frames to keep a track alive without detections
            initialization_delay: Frames to wait before initializing a new track
            frame_rate: Video frame rate
        """
        self.model = RFDETRBase(pretrain_weights=model_path)
        self.conf_threshold = conf_threshold
        self.frame_rate = frame_rate
        
        # Initialize Norfair tracker
        self.tracker = Tracker(
            distance_function="euclidean",  # Use centroid distance for bee tracking
            distance_threshold=distance_threshold,
            hit_counter_max=hit_counter_max,
            initialization_delay=initialization_delay,
            pointwise_hit_counter_max=hit_counter_max
        )
        
        # Track statistics
        self.track_history = defaultdict(list)  # track_id -> [frame_numbers]
        self.track_durations = {}  # track_id -> duration in seconds
        self.track_classes = defaultdict(list)  # track_id -> [class_ids for each detection]
        self.frame_count = 0
        
    def detect_frame(self, frame):
        """
        Run RF-DETR inference on a single frame
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            List of Norfair Detection objects
        """
        # Convert BGR to RGB for PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        try:
            # RF-DETR inference
            pred = self.model.predict(pil_image, threshold=self.conf_threshold)
            
            detections = []
            if len(pred.xyxy) > 0:
                for box, cls_id, conf in zip(pred.xyxy, pred.class_id, pred.confidence):
                    if conf >= self.conf_threshold:
                        x1, y1, x2, y2 = box
                        
                        # Calculate centroid for Norfair
                        centroid_x = (x1 + x2) / 2
                        centroid_y = (y1 + y2) / 2
                        
                        # Create Norfair Detection object
                        # points: centroid coordinates as numpy array
                        # scores: confidence scores
                        # data: additional data (bounding box, class)
                        detection = Detection(
                            points=np.array([[centroid_x, centroid_y]]),
                            scores=np.array([conf]),
                            data={
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'class_id': int(cls_id),
                                'confidence': float(conf)
                            }
                        )
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Error in detection: {e}")
            return []
    
    def update_tracker(self, detections, frame_idx):
        """
        Update tracker with new detections
        
        Args:
            detections: List of Norfair Detection objects
            frame_idx: Current frame index
            
        Returns:
            List of TrackedObject instances
        """
        # Update tracker
        tracked_objects = self.tracker.update(detections=detections)
        
        # Update track history and class information
        for tracked_obj in tracked_objects:
            track_id = tracked_obj.id
            if track_id is not None and tracked_obj.last_detection is not None:  # Only count initialized tracks
                self.track_history[track_id].append(frame_idx)
                # Store the class_id from the last detection
                class_id = tracked_obj.last_detection.data['class_id']
                self.track_classes[track_id].append(class_id)
        
        return tracked_objects
    
    def get_dominant_class(self, track_id):
        """
        Get the most common class for a track
        
        Args:
            track_id: Track ID
            
        Returns:
            Most common class_id for this track
        """
        if track_id not in self.track_classes or not self.track_classes[track_id]:
            return None
        
        # Count occurrences of each class
        class_counts = {}
        for class_id in self.track_classes[track_id]:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        # Return the most common class
        return max(class_counts, key=class_counts.get)
    
    def calculate_track_durations(self):
        """Calculate duration for each track ID"""
        for track_id, frame_list in self.track_history.items():
            if len(frame_list) > 0:
                start_frame = min(frame_list)
                end_frame = max(frame_list)
                duration_frames = end_frame - start_frame + 1
                duration_seconds = duration_frames / self.frame_rate
                self.track_durations[track_id] = duration_seconds
    
    def draw_tracks(self, frame, tracked_objects):
        """
        Draw bounding boxes and track IDs on frame
        
        Args:
            frame: OpenCV frame
            tracked_objects: List of TrackedObject instances from update_tracker
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        active_tracks = 0
        
        for tracked_obj in tracked_objects:
            if tracked_obj.id is not None and tracked_obj.last_detection is not None:
                active_tracks += 1
                
                # Get track info from last detection
                track_id = tracked_obj.id
                detection_data = tracked_obj.last_detection.data
                
                # Extract bounding box
                x1, y1, x2, y2 = detection_data['bbox']
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                conf = detection_data['confidence']
                class_id = detection_data['class_id']
                
                # Choose color based on track ID for consistency
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                         (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
                color = colors[track_id % len(colors)]
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw track ID, class, and confidence
                label = f"ID:{track_id} C:{class_id} ({conf:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Background for text
                cv2.rectangle(annotated_frame, 
                             (x1, y1 - label_size[1] - 5), 
                             (x1 + label_size[0], y1), 
                             color, -1)
                
                # Text
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw centroid
                centroid = tracked_obj.estimate
                if centroid is not None and len(centroid) > 0:
                    cx, cy = int(centroid[0][0]), int(centroid[0][1])
                    cv2.circle(annotated_frame, (cx, cy), 3, color, -1)
        
        # Add frame info
        info_text = f"Active Tracks: {active_tracks} | Frame: {self.frame_count}"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame

def process_video(video_path, output_path, model_path, conf_threshold=0.5, distance_threshold=30, hit_counter_max=10, initialization_delay=3):
    """
    Process video with bee detection and tracking using Norfair
    
    Args:
        video_path: Input video path
        output_path: Output video path
        model_path: RF-DETR model path
        conf_threshold: Detection confidence threshold
        distance_threshold: Maximum distance for matching tracks (pixels)
        hit_counter_max: Maximum frames to keep a track alive
        initialization_delay: Frames to wait before initializing new track
    """
    # Initialize tracker
    tracker = BeeTracker(model_path, conf_threshold, distance_threshold, hit_counter_max, initialization_delay)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    
    # Process video frame by frame
    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update tracker frame rate info
            tracker.frame_rate = fps
            tracker.frame_count = frame_idx
            
            # Detect objects in current frame
            detections = tracker.detect_frame(frame)
            
            # Update tracker
            tracked_objects = tracker.update_tracker(detections, frame_idx)
            
            # Draw tracks on frame
            annotated_frame = tracker.draw_tracks(frame, tracked_objects)
            
            # Write frame
            out.write(annotated_frame)
            
            frame_idx += 1
            pbar.update(1)
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Calculate track durations
    tracker.calculate_track_durations()
    
    # Print statistics
    print("\n" + "="*50)
    print("BEE TRACKING STATISTICS")
    print("="*50)
    print(f"Total number of unique bee tracks: {len(tracker.track_durations)}")
    print(f"Video duration: {total_frames/fps:.2f} seconds")
    print("\nTrack durations:")
    print("-"*40)
    
    # Sort tracks by duration (longest first)
    sorted_tracks = sorted(tracker.track_durations.items(), key=lambda x: x[1], reverse=True)
    
    total_track_time = 0
    for track_id, duration in sorted_tracks:
        frame_count = len(tracker.track_history[track_id])
        dominant_class = tracker.get_dominant_class(track_id)
        print(f"Track ID {track_id:3d}: {duration:6.2f}s ({frame_count:4d} frames) Class: {dominant_class}")
        total_track_time += duration
    
    print("-"*40)
    print(f"Total tracking time: {total_track_time:.2f} seconds")
    if len(tracker.track_durations) > 0:
        print(f"Average track duration: {total_track_time/len(tracker.track_durations):.2f} seconds")
        print(f"Longest track: {max(tracker.track_durations.values()):.2f} seconds")
        print(f"Shortest track: {min(tracker.track_durations.values()):.2f} seconds")
    
    # Save statistics to JSON file
    stats_path = output_path.replace('.mp4', '_stats.json')
    
    # Get video filename without path and extension for cleaner reference
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create track details with class information
    track_details = {}
    for track_id in tracker.track_durations.keys():
        dominant_class = tracker.get_dominant_class(track_id)
        track_details[str(track_id)] = {
            'duration_seconds': tracker.track_durations[track_id],
            'frame_count': len(tracker.track_history[track_id]),
            'dominant_class_id': dominant_class,
            'all_classes': list(set(tracker.track_classes[track_id])) if track_id in tracker.track_classes else []
        }
    
    stats = {
        'video_name': video_name,
        'video_path': video_path,
        'total_tracks': len(tracker.track_durations),
        'video_duration_seconds': total_frames/fps,
        'total_tracking_time_seconds': total_track_time,
        'average_track_duration_seconds': total_track_time/len(tracker.track_durations) if len(tracker.track_durations) > 0 else 0,
        'longest_track_seconds': max(tracker.track_durations.values()) if tracker.track_durations else 0,
        'shortest_track_seconds': min(tracker.track_durations.values()) if tracker.track_durations else 0,
        'track_details': track_details,
        # Legacy fields for backward compatibility
        'track_durations': {str(k): v for k, v in tracker.track_durations.items()},
        'track_frame_counts': {str(k): len(v) for k, v in tracker.track_history.items()}
    }
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStatistics saved to: {stats_path}")
    print(f"Annotated video saved to: {output_path}")

def process_multiple_videos(video_paths, output_dir, model_path, conf_threshold=0.5, distance_threshold=30, hit_counter_max=10, initialization_delay=3):
    """
    Process multiple videos with bee detection and tracking
    
    Args:
        video_paths: List of input video paths
        output_dir: Output directory for all results
        model_path: RF-DETR model path
        conf_threshold: Detection confidence threshold
        distance_threshold: Maximum distance for matching tracks (pixels)
        hit_counter_max: Maximum frames to keep a track alive
        initialization_delay: Frames to wait before initializing new track
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, video_path in enumerate(video_paths):
        print(f"\n{'='*60}")
        print(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
        print(f"{'='*60}")
        
        if not os.path.exists(video_path):
            print(f"Warning: Video file {video_path} does not exist. Skipping...")
            continue
        
        # Create output filename based on input filename
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{video_name}_tracked.mp4")
        
        # Process the video
        process_video(video_path, output_path, model_path, conf_threshold, distance_threshold, hit_counter_max, initialization_delay)

def main():
    # Configuration parameters - modify these as needed
    video_path = "/scratch/s52melba/test_vid.mov"  # Your input video path
    output_path = "/scratch/s52melba/out_tracking/test_tracked_video.mp4"  # Your desired output path
    model_path = "/scratch/s52melba/detr_train_2/checkpoint_best_total.pth"
    
    # Detection and tracking parameters
    conf_threshold = 0.3         # Detection confidence threshold
    distance_threshold = 90      # Max pixel distance for track matching (lower = stricter)
    hit_counter_max = 10         # Max frames to keep track alive without detection
    initialization_delay = 5     # Frames to wait before confirming new track
    
    # Example for processing multiple videos
    # video_paths = [
    #     "/path/to/video1.mov",
    #     "/path/to/video2.mp4",
    #     "/path/to/video3.avi"
    # ]
    # output_dir = "/scratch/s52melba/batch_output"
    # process_multiple_videos(video_paths, output_dir, model_path, conf_threshold, distance_threshold, hit_counter_max, initialization_delay)
    
    # Check if input video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist")
        print("Please modify the 'video_path' variable in the main() function")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Starting bee tracking with Norfair...")
    print(f"Input: {video_path}")
    print(f"Output: {output_path}")
    print(f"Detection threshold: {conf_threshold}")
    print(f"Distance threshold: {distance_threshold} pixels")
    print(f"Max missing frames: {hit_counter_max}")
    
    # Process video
    process_video(video_path, output_path, model_path, conf_threshold, distance_threshold, hit_counter_max, initialization_delay)

if __name__ == "__main__":
    main()

# Installation:
# pip install norfair
#
# Usage:
# 1. For single video: Modify the paths in the main() function and run
# 2. For multiple videos: Use the process_multiple_videos function with a list of video paths
# 
# Example for multiple videos:
# video_paths = ["/path/to/video1.mov", "/path/to/video2.mp4"]
# output_dir = "/output/directory"
# process_multiple_videos(video_paths, output_dir, model_path)
from sahi.slicing import slice_coco

coco_dict, coco_path = slice_coco(
    coco_annotation_file_path='/scratch/s52melba/coco_dataset/labels.json',
    image_dir='/scratch/s52melba/coco_dataset/data',
    output_dir='/scratch/s52melba/coco_sahi',
    output_coco_annotation_file_name='coco_sahi.json',
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,   
    )
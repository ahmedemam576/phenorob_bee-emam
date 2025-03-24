from sahi.slicing import slice_coco

coco_dict, coco_path = slice_coco(
    coco_annotation_file_path='/scratch/s52melba/coco_dataset/labels.json',
    image_dir='/scratch/s52melba/coco_dataset/data',
    output_dir='/scratch/s52melba/coco_sahi_256',
    output_coco_annotation_file_name='coco_sahi', # .json will be added automatically
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.4,
    overlap_width_ratio=0.4,   
 
    )
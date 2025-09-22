import json
import tqdm
import os

# path of COCO Dataset
coco_annotation_path = "datasets/coco/captions_val2017.json"
coco_images_path = "/workspace/data/coco/val2017"


output_path = "datasets/coco_split/"
os.makedirs(output_path, exist_ok=True)

# 拆分数量
split_number = 8

# 存储唯一的图像ID和对应的标注信息
unique_images = {}

# 读取 COCO 注释文件
with open(coco_annotation_path, 'r') as coco_annotation_file:
    coco_annotation = json.load(coco_annotation_file)
    for annotation in tqdm.tqdm(coco_annotation['annotations']):
        image_id = str(annotation['image_id']).zfill(12)
        image_path = str(annotation['image_id']).zfill(12) + ".jpg"
        image_path = os.path.join(coco_images_path, image_path)
        prompt = annotation['caption']
        
        if image_id not in unique_images:
            unique_images[image_id] = {
                'imageid': image_id,
                'image_path': image_path,
                'prompt': prompt
            }

# 转换为列表
unique_images_list = list(unique_images.values())
dataset_total_len = len(unique_images_list)
split_len = dataset_total_len // split_number

# 拆分并保存数据
for split_index in range(split_number):
    start_idx = split_index * split_len
    end_idx = (split_index + 1) * split_len if split_index < split_number - 1 else dataset_total_len
    split_data = unique_images_list[start_idx:end_idx]
    
    split_path = os.path.join(output_path, f"split_{split_index}.json")
    with open(split_path, 'w') as f:
        json.dump(split_data, f, indent=4)
    
    print(f"Saved {len(split_data)} samples to {split_path}")

total_split_path = os.path.join(output_path, f"total.json")
with open(total_split_path, 'w') as f:
    json.dump(unique_images_list, f, indent=4)

print(f"Saved {len(split_data)} samples to {total_split_path}")



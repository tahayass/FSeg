import os
import json


def convert_coco_to_yolo(coco_json, output_dir):
    with open(coco_json, 'r') as f:
        data = json.load(f)

    # Check if the annotations key exists
    if "annotations" not in data:
        raise ValueError("annotations key missing from the JSON file.")

    # Create a dictionary with image id as key and file_name as value
    images = {image['id']: image for image in data['images']}

    # Create a dictionary for categories
    category_id_to_name = {category['id']: category['name']
                           for category in data['categories']}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert annotations
    for ann in data['annotations']:
        image_id = ann['image_id']
        image = images[image_id]
        file_name = os.path.join(output_dir, os.path.splitext(
            image['file_name'])[0] + '.txt')

        # Calculate normalized width, height, center x, center y
        x1, y1, w, h = ann['bbox']
        center_x = (x1 + w / 2) / float(image['width'])
        center_y = (y1 + h / 2) / float(image['height'])
        w_norm = w / float(image['width'])
        h_norm = h / float(image['height'])

        # Convert category name to id (here you might need to adjust based on your own class mapping)
        category_id = ann['category_id']

        # Write to output file
        with open(file_name, 'a') as f:
            f.write(f"{category_id} {center_x} {center_y} {w_norm} {h_norm}\n")
            


if __name__ == '__main__':
    convert_coco_to_yolo(
        './Combined_dataset/train/_annotations.coco.json',
        './Combined_dataset/train/labels')
    convert_coco_to_yolo(
        './Combined_dataset/test/_annotations.coco.json',
        './Combined_dataset/test/labels')


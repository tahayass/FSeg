import json
import os
import shutil
import uuid
import random
from pycocotools.coco import COCO

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_example_images_with_bboxes(coco_json_file, image_dir, num_images=5):

    coco = COCO(coco_json_file)

    image_ids = coco.getImgIds()
    random_sample = random.sample(image_ids, num_images)

    # Load the COCO dataset from the JSON file
    with open(coco_json_file, 'r') as f:
        coco_data = json.load(f)


    for idx in random_sample:
        image_info = coco.loadImgs(idx)[0]
        image_path = os.path.join(image_dir, image_info['file_name'])
        # Load and display the image using Matplotlib
        img = plt.imread(image_path)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis('off')

        # Get annotations for this image
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=idx))
        
        # Draw bounding boxes and labels
        for ann in annotations:
            bbox = ann['bbox']
            category_id = ann['category_id']
            category_info = next(cat for cat in coco_data['categories'] if cat['id'] == category_id)
            label = category_info['name']

            x, y, width, height = bbox
            rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
            plt.text(x, y - 5, label, color='r', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

        plt.show()


def merge_datasets(input_dirs, output_dir):
    # Initialize the merged COCO dataset
    merged_coco = {
        "info": {},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }

    # Create a mapping to reindex category IDs
    category_id_mapping = {}

    image_id_counter = 0
    annotation_id_counter = 0
    category_names=[]
    old_id=0

    for input_dir in input_dirs:
        annotation_file = os.path.join(input_dir,"_annotations.coco.json")
        with open(annotation_file, "r") as f:
            dataset = json.load(f)
        # Update category IDs and add them to the merged dataset
        for category in dataset["categories"]:
            if (category["name"] in category_names)==False:
                merged_coco["categories"].append(category)
                category_id_mapping[category["id"]] = len(merged_coco["categories"]) + 1
                category["id"] = old_id 
                category_names.append(category["name"])
                old_id += 1

    
    for input_dir in input_dirs:
        # Load the COCO annotation file
        annotation_file = os.path.join(input_dir,"_annotations.coco.json")
        coco = COCO(annotation_file)
        with open(annotation_file, "r") as f:
            dataset = json.load(f)

        original_categories = dataset["categories"]

        for image in dataset["images"]:
                
            unique_filename = str(uuid.uuid4()) + os.path.splitext(image["file_name"])[-1]
            output_image_path = os.path.join(output_dir, unique_filename)

            # Copy the image to the output directory
            shutil.copy(os.path.join(input_dir, image["file_name"]), output_image_path)
            original_id = image["id"]
            image_annotations = coco.loadAnns(coco.getAnnIds(imgIds=original_id))
            # Update image information
            image["id"] = image_id_counter
            image["file_name"] = unique_filename
            image_id_counter += 1
            merged_coco["images"].append(image)
            
            # Add images and annotations from the current dataset to the merged dataset
            for  annotation in image_annotations:
                annotation["id"] = annotation_id_counter
                annotation["image_id"] = image["id"]
                for cat in original_categories:
                    if cat['id'] == annotation["category_id"]:
                        category_name = cat['name']
                for cat in merged_coco["categories"]:
                    if cat['name'] == category_name:
                        new_ann_id = cat['id']

                annotation["category_id"] = new_ann_id
                annotation_id_counter += 1
                merged_coco["annotations"].append(annotation)

    # Save the merged COCO dataset
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "_annotations.coco.json")
    with open(output_file, "w") as f:
        json.dump(merged_coco, f)


if __name__ == "__main__":

    input_dirs = [r".\Data\Potato Pancake.v2i.coco",
    r".\Data\Leaf Mustard Kimchi, Radish kimchi, Julienne Radish Fresh Salad, Kimchi, White Kimchi, Chive Kimchi.v1i.coco",
    r".\Data\Pickled Perilla Leaf, Braised Beans, Stir-fried Eggplant, Bracken Salad.v5i.coco"]
    output_dir = ".\Data\combined_dataset2"

    if os.path.exists(output_dir)==False:
        os.mkdir(output_dir)
  
    test_dirs = [os.path.join(input_dir,"test") for input_dir in input_dirs]
    os.mkdir(os.path.join(output_dir,"test"))
    merge_datasets(test_dirs, os.path.join(output_dir,"test"))
    train_dirs = [os.path.join(input_dir,"test") for input_dir in input_dirs]
    os.mkdir(os.path.join(output_dir,"train"))
    merge_datasets(train_dirs, os.path.join(output_dir,"train"))

    show_example_images_with_bboxes(os.path.join(output_dir,"test","_annotations.coco.json"), os.path.join(output_dir,"test"), num_images=5)

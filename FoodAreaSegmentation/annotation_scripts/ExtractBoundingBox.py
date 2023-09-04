import json
from pycocotools.coco import COCO

def load_coco_data(annotation_file):
    # Initialize COCO instance
    coco = COCO(annotation_file)

    # Get all image IDs in the dataset
    image_ids = coco.getImgIds()

    # Initialize lists to store images, annotations, and category IDs
    images = []
    annotations = []
    category_id_to_name = {}  # Create a dictionary for category ID to name mapping

    # Load category information
    categories_info = coco.loadCats(coco.getCatIds())
    for category_info in categories_info:
        category_id = category_info['id']
        category_name = category_info['name']
        category_id_to_name[category_id] = category_name  # Add to the mapping dictionary

    # Loop through image IDs
    for image_id in image_ids:
        # Load image information
        image_info = coco.loadImgs(image_id)[0]

        # Load annotations for the current image
        image_annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))

        # Append image and annotations to respective lists
        images.append(image_info)
        annotations.extend(image_annotations)

    return images, annotations, category_id_to_name

if __name__ == "__main__":
    # Example usage:
    annotation_file_path = r'.\Data\Fseg-food-detection-1\test\_annotations.coco.json'
    images, annotations, categories = load_coco_data(annotation_file_path)
    print(images)

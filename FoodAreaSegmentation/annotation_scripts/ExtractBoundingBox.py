from pycocotools.coco import COCO

def load_coco_data(annotation_file):

    # Initialize COCO instance
    coco = COCO(annotation_file)

    # Create a dictionary for category ID to name mapping
    category_id_to_name = {} 

    # Load category information
    categories_info = coco.loadCats(coco.getCatIds())
    for category_info in categories_info:
        category_id = category_info['id']
        category_name = category_info['name']
        category_id_to_name[category_id] = category_name  

    return coco , category_id_to_name


def get_bounding_boxes_for_all_image(coco,category_id_to_name):


    # Get all image IDs in the dataset
    image_ids = coco.getImgIds()

    image_data = {}  # Maps image IDs to a list of annotations

    for image_id in image_ids:
        # Load image information
        image_info = coco.loadImgs(image_id)[0]

        # Load annotations for the current image
        image_annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))

        # Initialize a list to store annotations for this image
        annotations_for_image = []

        for annotation in image_annotations:
            category_id = annotation['category_id']
            category_name = category_id_to_name.get(category_id, 'Unknown')  # Get category name

            # Extract bounding box coordinates
            bbox = annotation['bbox']

            # Create a dictionary for the annotation
            annotation_data = {
                'category_id': category_id,
                'category_name': category_name,
                'bbox': bbox
            }

            annotations_for_image.append(annotation_data)

        # Add the list of annotations for this image to the image_data dictionary
        image_data[image_id] = annotations_for_image

    return image_data


def get_image_info(coco, image_id):
        
    # image information for the specified image_id
    image_info = coco.loadImgs(image_id)[0]
    return image_info


if __name__ == "__main__":
    # Example usage:
    annotation_file_path = r'.\Data\Fseg-food-detection-1\test\_annotations.coco.json'
    coco, category_id_to_name = load_coco_data(annotation_file_path)
    print(category_id_to_name)
    image_data = get_bounding_boxes_for_all_image(coco,category_id_to_name)
    image_id = 0
    print(image_data[image_id])
    
    



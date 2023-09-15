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
        './Dataset/Braised Bean Curd, Braised Potatoes, Pan-fried Battered Zucchini.v10i.coco/test/_annotations.coco.json',
        './Dataset/Braised Bean Curd, Braised Potatoes, Pan-fried Battered Zucchini.v10i.coco/test/labels')
    convert_coco_to_yolo(
        './Dataset/Braised Peanuts, Braised Lotus Roots.v11i.coco/train/_annotations.coco.json',
        './Dataset/Braised Peanuts, Braised Lotus Roots.v11i.coco/train/labels')
    convert_coco_to_yolo(
        './Dataset/Braised Peanuts, Braised Lotus Roots.v11i.coco/test/_annotations.coco.json',
        './Dataset/Braised Peanuts, Braised Lotus Roots.v11i.coco/test/labels')
    convert_coco_to_yolo(
        './Dataset/Bulgogi, Grilled Pork Belly, Grilled Eel, Grilled Clams, grilled yellow corvina, smoked duck, Seasoned Fried Chicken, Pizza.v17i.coco/train/_annotations.coco.json',
        './Dataset/Bulgogi, Grilled Pork Belly, Grilled Eel, Grilled Clams, grilled yellow corvina, smoked duck, Seasoned Fried Chicken, Pizza.v17i.coco/train/labels')
    convert_coco_to_yolo(
        './Dataset/Bulgogi, Grilled Pork Belly, Grilled Eel, Grilled Clams, grilled yellow corvina, smoked duck, Seasoned Fried Chicken, Pizza.v17i.coco/test/_annotations.coco.json',
        './Dataset/Bulgogi, Grilled Pork Belly, Grilled Eel, Grilled Clams, grilled yellow corvina, smoked duck, Seasoned Fried Chicken, Pizza.v17i.coco/test/labels')
    convert_coco_to_yolo(
        './Dataset/Cucumber Kimchi.v3i.coco/train/_annotations.coco.json',
        './Dataset/Cucumber Kimchi.v3i.coco/train/labels')
    convert_coco_to_yolo(
        './Dataset/Cucumber Kimchi.v3i.coco/test/_annotations.coco.json',
        './Dataset/Cucumber Kimchi.v3i.coco/test/labels')
    convert_coco_to_yolo(
        './Dataset/Eoyonam\'s Stir-fried dried squid with Red Chili Paste, Stir-fried Rice Cake (non noodle), Stir-fried Rice Cake, Rice Cake Skewers.v8i.coco/train/_annotations.coco.json',
        './Dataset/Eoyonam\'s Stir-fried dried squid with Red Chili Paste, Stir-fried Rice Cake (non noodle), Stir-fried Rice Cake, Rice Cake Skewers.v8i.coco/train/labels')
    convert_coco_to_yolo(
        './Dataset/Eoyonam\'s Stir-fried dried squid with Red Chili Paste, Stir-fried Rice Cake (non noodle), Stir-fried Rice Cake, Rice Cake Skewers.v8i.coco/test/_annotations.coco.json',
        './Dataset/Eoyonam\'s Stir-fried dried squid with Red Chili Paste, Stir-fried Rice Cake (non noodle), Stir-fried Rice Cake, Rice Cake Skewers.v8i.coco/test/labels')
    convert_coco_to_yolo(
        './Dataset/Grilled Cutlassfish, Grilled Mackerel, Grilled Beef TripeGrilled Pork Tripe.v15i.coco/train/_annotations.coco.json',
        './Dataset/Grilled Cutlassfish, Grilled Mackerel, Grilled Beef TripeGrilled Pork Tripe.v15i.coco/train/labels')
    convert_coco_to_yolo(
        './Dataset/Grilled Cutlassfish, Grilled Mackerel, Grilled Beef TripeGrilled Pork Tripe.v15i.coco/test/_annotations.coco.json',
        './Dataset/Grilled Cutlassfish, Grilled Mackerel, Grilled Beef TripeGrilled Pork Tripe.v15i.coco/test/labels')
    convert_coco_to_yolo(
        './Dataset/Kimchi Pancake, Green Onion Pancake.v9i.coco/train/_annotations.coco.json',
        './Dataset/Kimchi Pancake, Green Onion Pancake.v9i.coco/train/labels')
    convert_coco_to_yolo(
        './Dataset/Kimchi Pancake, Green Onion Pancake.v9i.coco/test/_annotations.coco.json',
        './Dataset/Kimchi Pancake, Green Onion Pancake.v9i.coco/test/labels')
    convert_coco_to_yolo(
        './Dataset/Kimchi Pancake, Green Onion Pancake.v9i.coco/train/_annotations.coco.json',
        './Dataset/Kimchi Pancake, Green Onion Pancake.v9i.coco/train/labels')
    convert_coco_to_yolo(
        './Dataset/Leaf Mustard Kimchi, Radish kimchi, Julienne Radish Fresh Salad, Kimchi, White Kimchi, Chive Kimchi.v1i.coco/train/_annotations.coco.json',
        './Dataset/Leaf Mustard Kimchi, Radish kimchi, Julienne Radish Fresh Salad, Kimchi, White Kimchi, Chive Kimchi.v1i.coco/train/labels')
    convert_coco_to_yolo(
        './Dataset/Leaf Mustard Kimchi, Radish kimchi, Julienne Radish Fresh Salad, Kimchi, White Kimchi, Chive Kimchi.v1i.coco/test/_annotations.coco.json',
        './Dataset/Leaf Mustard Kimchi, Radish kimchi, Julienne Radish Fresh Salad, Kimchi, White Kimchi, Chive Kimchi.v1i.coco/test/labels')
    convert_coco_to_yolo(
        './Dataset/Pickled Perilla Leaf, Braised Beans, Stir-fried Eggplant, Bracken Salad.v5i.coco/train/_annotations.coco.json',
        './Dataset/Pickled Perilla Leaf, Braised Beans, Stir-fried Eggplant, Bracken Salad.v5i.coco/train/labels')
    convert_coco_to_yolo(
        './Dataset/Pickled Perilla Leaf, Braised Beans, Stir-fried Eggplant, Bracken Salad.v5i.coco/test/_annotations.coco.json',
        './Dataset/Pickled Perilla Leaf, Braised Beans, Stir-fried Eggplant, Bracken Salad.v5i.coco/test/labels')
    convert_coco_to_yolo(
        './Dataset/Potato Pancake.v2i.coco/train/_annotations.coco.json',
        './Dataset/Potato Pancake.v2i.coco/train/labels')
    convert_coco_to_yolo(
        './Dataset/Potato Pancake.v2i.coco/test/_annotations.coco.json',
        './Dataset/Potato Pancake.v2i.coco/test/labels')
    convert_coco_to_yolo(
        './Dataset/Spicy Stir-fried Chicken, Grilled Ribs, Grilled Short Rib Patties.v16i.coco/train/_annotations.coco.json',
        './Dataset/Spicy Stir-fried Chicken, Grilled Ribs, Grilled Short Rib Patties.v16i.coco/train/labels')
    convert_coco_to_yolo(
        './Dataset/Spicy Stir-fried Chicken, Grilled Ribs, Grilled Short Rib Patties.v16i.coco/test/_annotations.coco.json',
        './Dataset/Spicy Stir-fried Chicken, Grilled Ribs, Grilled Short Rib Patties.v16i.coco/test/labels')
    convert_coco_to_yolo(
        './Dataset/Stir-fried Seaweed Stems, Mung Bean Sprout Salad, Spinach Salad, Stir-fried Zucchini.v6i.coco/train/_annotations.coco.json',
        './Dataset/Stir-fried Seaweed Stems, Mung Bean Sprout Salad, Spinach Salad, Stir-fried Zucchini.v6i.coco/train/labels')
    convert_coco_to_yolo(
        './Dataset/Stir-fried Seaweed Stems, Mung Bean Sprout Salad, Spinach Salad, Stir-fried Zucchini.v6i.coco/test/_annotations.coco.json',
        './Dataset/Stir-fried Seaweed Stems, Mung Bean Sprout Salad, Spinach Salad, Stir-fried Zucchini.v6i.coco/test/labels')
    convert_coco_to_yolo(
        './Dataset/Stir-fried Shishito Peppers, seasoned balloon flower roots, Acorn Jelly Salad, Bean Sprout Salad, Stir-fried Potatoes.v7i.coco/train/_annotations.coco.json',
        './Dataset/Stir-fried Shishito Peppers, seasoned balloon flower roots, Acorn Jelly Salad, Bean Sprout Salad, Stir-fried Potatoes.v7i.coco/train/labels')
    convert_coco_to_yolo(
        './Dataset/Stir-fried Shishito Peppers, seasoned balloon flower roots, Acorn Jelly Salad, Bean Sprout Salad, Stir-fried Potatoes.v7i.coco/test/_annotations.coco.json',
        './Dataset/Stir-fried Shishito Peppers, seasoned balloon flower roots, Acorn Jelly Salad, Bean Sprout Salad, Stir-fried Potatoes.v7i.coco/test/labels')
    convert_coco_to_yolo(
        './Dataset/Whole Radish Kimchi.v4i.coco/train/_annotations.coco.json',
        './Dataset/Whole Radish Kimchi.v4i.coco/train/labels')
    convert_coco_to_yolo(
        './Dataset/Whole Radish Kimchi.v4i.coco/test/_annotations.coco.json',
        './Dataset/Whole Radish Kimchi.v4i.coco/test/labels')

import os
import random
import shutil

# Define paths to your original dataset
dataset_root = 'Combined_dataset2'
images_dir = os.path.join(dataset_root, 'images')
labels_dir = os.path.join(dataset_root, 'labels')
train_images_dir = os.path.join(images_dir, 'train')
train_labels_dir = os.path.join(labels_dir, 'train')
test_images_dir = os.path.join(images_dir, 'test')
test_labels_dir = os.path.join(labels_dir, 'test')

# Desired number of images per class in train and test sets
desired_images_per_class_train = 150
desired_images_per_class_test = 20

# Define the path to the new dataset folder
new_dataset_root = 'Combined_dataset_reduced'
new_images_dir = os.path.join(new_dataset_root, 'images')
new_labels_dir = os.path.join(new_dataset_root, 'labels')
new_train_images_dir = os.path.join(new_images_dir, 'train')
new_train_labels_dir = os.path.join(new_labels_dir, 'train')
new_test_images_dir = os.path.join(new_images_dir, 'test')
new_test_labels_dir = os.path.join(new_labels_dir, 'test')

# Create the new dataset folder structure
os.makedirs(new_train_images_dir, exist_ok=True)
os.makedirs(new_train_labels_dir, exist_ok=True)
os.makedirs(new_test_images_dir, exist_ok=True)
os.makedirs(new_test_labels_dir, exist_ok=True)

# Create a backup directory to move excess images (optional)
backup_dir = 'backup_excess_images'
os.makedirs(backup_dir, exist_ok=True)

# Define the list of supported image extensions
supported_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

# Count images per class in the train set
class_image_counts_train = {}
for label_file in os.listdir(train_labels_dir):
    with open(os.path.join(train_labels_dir, label_file), 'r') as f:
        lines = f.readlines()
        class_id_list = [line.split()[0] for line in lines]
        class_id_set = set(class_id_list)
        for class_id in class_id_set:
            class_image_counts_train[class_id] = class_image_counts_train.get(
                class_id, 0) + class_id_list.count(class_id)

# Process each class in the train set
for class_id, count in class_image_counts_train.items():
    if count > desired_images_per_class_train:
        # Calculate the number of excess images
        excess_count = count - desired_images_per_class_train
        print(f"Train Set - Class {class_id}: Excess Images = {excess_count}")

        # List all image files for the class in the train set with supported extensions
        class_image_files = [f for f in os.listdir(train_images_dir) if os.path.splitext(f)[1].lower(
        ) in supported_image_extensions and f.replace(os.path.splitext(f)[1], '.txt') in os.listdir(train_labels_dir)]
        # Shuffle the list to randomly select images
        random.shuffle(class_image_files)

        # Move or delete excess images in the train set
        for i in range(excess_count):
            image_file = class_image_files[i]
            label_file = image_file.replace(
                os.path.splitext(image_file)[1], '.txt')
            image_path = os.path.join(train_images_dir, image_file)
            label_path = os.path.join(train_labels_dir, label_file)

            # Create the same folder structure in the new dataset folder for the train set
            new_image_path = os.path.join(new_train_images_dir, image_file)
            new_label_path = os.path.join(new_train_labels_dir, label_file)

            if os.path.exists(image_path):
                shutil.move(image_path, new_image_path)
            if os.path.exists(label_path):
                shutil.move(label_path, new_label_path)

# Count images per class in the test set
class_image_counts_test = {}
for label_file in os.listdir(test_labels_dir):
    with open(os.path.join(test_labels_dir, label_file), 'r') as f:
        lines = f.readlines()
        class_id_list = [line.split()[0] for line in lines]
        class_id_set = set(class_id_list)
        for class_id in class_id_set:
            class_image_counts_test[class_id] = class_image_counts_test.get(
                class_id, 0) + class_id_list.count(class_id)

# Process each class in the test set
for class_id, count in class_image_counts_test.items():
    if count > desired_images_per_class_test:
        # Calculate the number of excess images
        excess_count = count - desired_images_per_class_test
        print(f"Test Set - Class {class_id}: Excess Images = {excess_count}")

        # List all image files for the class in the test set with supported extensions
        class_image_files = [f for f in os.listdir(test_images_dir) if os.path.splitext(f)[1].lower(
        ) in supported_image_extensions and f.replace(os.path.splitext(f)[1], '.txt') in os.listdir(test_labels_dir)]
        # Shuffle the list to randomly select images
        random.shuffle(class_image_files)

        # Move or delete excess images in the test set
        for i in range(excess_count):
            image_file = class_image_files[i]
            label_file = image_file.replace(
                os.path.splitext(image_file)[1], '.txt')
            image_path = os.path.join(test_images_dir, image_file)
            label_path = os.path.join(test_labels_dir, label_file)

            # Create the same folder structure in the new dataset folder for the test set
            new_image_path = os.path.join(new_test_images_dir, image_file)
            new_label_path = os.path.join(new_test_labels_dir, label_file)

            if os.path.exists(image_path):
                shutil.move(image_path, new_image_path)
            if os.path.exists(label_path):
                shutil.move(label_path, new_label_path)

print("Processing completed. New dataset saved in:", new_dataset_root)

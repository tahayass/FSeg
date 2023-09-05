# Fseg - Solution for Food Detection, Segmentation, and Nutrition Value Calculation

## Table of Contents
- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

## Description

The challenge at hand is to develop a comprehensive solution for automating the detection, segmentation, classification of food items within images, and subsequently calculating their nutrition values.

## Features

Our proposed solution is a multi-step process that involves utilizing computer vision techniques and deep learning models to achieve accurate food detection, segmentation, and nutrition value calculation.

### Step 1: Plate Detection and Circular Extraction

We begin by detecting the presence of a plate within the image. This is achieved through bounding box detection, followed by precise circular extraction to identify the boundaries of the plate. This step sets the foundation for isolating the food items from the background.


### Step 2: Food Type Detection and Classification

Once the plate is identified, we move on to the crucial task of food type detection and classification. For this purpose, we employ the YOLO (You Only Look Once) algorithm, a deep learning model renowned for its real-time object detection capabilities. YOLO accurately identifies and classifies different food items present on the plate, providing a detailed inventory of the meal composition.


### Step 3: Food Area Segmentation

Moving forward from food type detection, our solution integrates the cutting-edge SAM (Segment Anything) model developed by Meta. SAM is an open-source framework renowned for its exceptional object segmentation capabilities. The unique advantage of SAM is that it doesn't necessitate manual annotations of specific objects, in this case, food items. The model is pre-trained and is designed to work across diverse scenarios, making it an ideal fit for our segmentation needs.

By employing SAM, we can accurately segment the food areas within the bounding boxes obtained from Step 2. This segmentation process effectively separates each food item from its surroundings, regardless of the variations in shape, size, or appearance. SAM's proficiency in handling complex and varied object structures enhances the precision of our solution.


### Step 4: Nutrition Value Calculation

Using the pixel count of the segmentation mask and knowing the dimensions of the plate, we can calculate the exact area of each type of food. By referencing established nutritional databases, we assign caloric, macronutrient, and micronutrient values to every recognized food item. This step enables users to access detailed nutritional insights for informed dietary decisions.

## Installation


## Usage



## Example


## Contributing


## License


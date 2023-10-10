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

To get started with our project, follow these steps:

1. **Clone the Repository**: Start by cloning this GitHub repository to your local machine using the following command:

    ```bash
    git clone https://github.com/tahayass/FSeg.git
    ```

2. **Navigate to the Project Directory**: Change your current directory to the project's root directory:

    ```bash
    cd FSeg
    ```

3. **Create a Conda Environment**: We recommend using [Anaconda](https://www.anaconda.com/) for managing Python environments. If you don't have Anaconda installed, you can download and install it from their website.

4. **Set Up the Conda Environment**: Create a Conda environment and install the required dependencies listed in the `environment.yml` file. This file specifies all the necessary packages and their versions for our project:

    ```bash
    conda env create -f environment.yml
    ```

   This command will create a Conda environment with the name specified in `environment.yml` and install all the required packages.

5. **Activate the Conda Environment**: Activate the Conda environment to work within it:

    ```bash
    conda activate FsegEnv
    ```


That's it! You've successfully set up the project on your local machine using Conda. 



## Usage

To use our project for food detection, follow these steps:

1. **Run the Food Detection Script**: Open your terminal and navigate to the project directory if you're not already there. Then, use the following command to run the food detection script:

    ```bash
    python main.py --weights PlateDetection/latest_food_detection_model --source path/to/your/test_image.jpg
    ```

   - `--weights PlateDetection/latest_food_detection_model`: This part of the command specifies the path to the latest model uploaded for food detection. You should replace `PlateDetection/latest_food_detection_model` with the actual path to the model file you want to use.
   
   - `--source path/to/your/test_image.jpg`: Here, you should specify the path to the image you want to test for food detection. Replace `path/to/your/test_image.jpg` with the actual path to your test image.

2. **Retrieve Results**: After running the command, relevant information about the detected food items, their locations, and any other details will be displayed in the console. Additionally, the results will be stored in the "PipelineTestResults" directory within your project.


That's it! You've successfully used our project to perform food detection on your test image.




## Example

Here's an example image generated by our food detection pipeline:

![Example Image](PipelineTestResults/test.jpg)

Console output as a dictionnary with food types and their respective pixel count : {'Julienne Radish Fresh Salad': 41331}

In this image, you can see the detections of various food items, each outlined by bounding boxes and labeled with their respective names and confidence scores.

This example demonstrates the visual output produced by our pipeline. The actual output may vary depending on the test image and model used.



## Contributing


## License


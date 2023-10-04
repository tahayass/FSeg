import torch
import os
import cv2
import numpy as np

from FoodAreaSegmentation.sam_model import GenerateMaskForImage


def get_plate_placement(image,
                        model,
                        resize=None):
    return None

def remove_background(image,
                      center,
                      diameter):
    return None

def get_food_bboxes(image,
                    model,
                    resize=None):
    bboxes = None
    food_types = None 

    return bboxes,food_types

def get_food_masks(image,
                   bboxes,
                   open=True,
                   close=True,
                   kernel_size=None):
    
    masks,_ = GenerateMaskForImage(image, bounding_boxes=bboxes,open=open,close=close,kernel_size=kernel_size)
    
    return masks

def calculate_surface_area(image,
                           masks,
                           bboxes,
                           food_types):
    # Create a dictionary to store the sums of masks with the same name
    mask_dict = {}

    # Iterate over each mask and its corresponding name
    for mask, name in zip(masks, food_types):
        if name not in mask_dict:
            mask_dict[name] = mask*1
        else:
            mask_dict[name] += mask*1

    # Create a dictionary to store the count of ones in each array
    pixel_count = {}

    # Iterate over the dictionary items and sum in each mask
    for name, summed_mask in mask_dict.items():
        non_zero_count = np.sum(summed_mask)
        pixel_count[name] = non_zero_count

    return pixel_count



if __name__=='__main__':
    #Constant variables
    SAM_CHECKPOINT = os.path.join('.','FoodAreaSegmentation','sam_vit_h_4b8939.pth')
    MODEL_TYPE = "vit_h"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #load plate detection model
    plate_detection_model = None

    #load plate detection model
    food_detection_model = None

    #Load image
    image_path = r''
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #Detect plate
    center,diameter = get_plate_placement(image,
                        plate_detection_model,
                        resize=None)
    
    #Remove background of the plate
    image = remove_background(image,
                      center,
                      diameter)
    
    #Detect food and classify them
    bboxes,food_types = get_food_bboxes(image,
                    food_detection_model,
                    resize=None)
    
    #Outputs segmentation mask for every food type
    masks = get_food_masks(image,
                   bboxes,
                   open=True,
                   close=True,
                   kernel_size=None)
    
    #Calculates masks pixel count and returns a dictionnary with surface area for every food {'food_type':pixel_count}
    areas = calculate_surface_area(image,
                                   masks,
                                   bboxes,
                                   food_types)
    

    




import sys 
import os

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from postprocessing import close_mask,open_mask
from FoodAreaSegmentation.utils import format_bbox



import numpy as np
import torch
import os

SAM_CHECKPOINT = os.path.join('.','FoodAreaSegmentation','sam_vit_h_4b8939.pth')
MODEL_TYPE = "vit_h"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def downloadSAMModel():
    return None


def LoadSAMPredictor(sam_checkpoint, model_type, device='cuda', return_sam=False):
    """
    Load a Segment Anything Model (SAM) predictor model for semantic segmentation.

    Parameters:
    - sam_checkpoint (str): The path to the checkpoint file containing the SAM model's weights and configuration.
    - model_type (str): The SAM model type to use. It should be a key that corresponds to a model in the 'sam_model_registry'.
    - device (str, optional): The device to run the model on, either 'cuda' (GPU) or 'cpu' (CPU). Default is 'cuda'.

    Returns:
    - predictor (SamPredictor): An instance of the SAM predictor configured with the specified model type and loaded weights.
    """
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    if return_sam :
        return predictor,sam
    else : 
        return predictor


def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.permute(2, 0, 1).contiguous()


def GenerateMask(predictor, input_box, include_point=True):
    """
    Generate segmentation masks for a given input bounding box using a SAM predictor.

    Parameters:
    - predictor (SamPredictor): An instance of the SAM predictor loaded with a model.
    - input_box (list or numpy array): The bounding box coordinates in the format [x_min, y_min, x_max, y_max].

    Returns:
    - masks (numpy array): A segmentation mask representing the predicted object(s) within the input bounding box.
    """
    if include_point:
        point_coords = np.array([[(input_box[0]+input_box[2])/2,(input_box[1]+input_box[3])/2]])
        point_labels=np.array([1]) #label 0 is for background and label 1 is for foreground
    else: 
        point_coords = None
        point_labels = None

    masks, iou, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,  
        box=input_box[None, :],
        multimask_output=False,
    )
    return masks, iou


def GenerateMaskBatch(sam, images, input_boxes, include_point=True):

    for i in range(len(input_boxes)):
        input_boxes[i] = torch.tensor(input_boxes[i], device=sam.device)

    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    batched_input = [
     {
         'image': prepare_image(images[0], resize_transform, sam),
         'boxes': resize_transform.apply_boxes_torch(input_boxes[0], images[0].shape[:2]),
         'original_size': images[0].shape[:2]
     },
     {
         'image': prepare_image(images[1], resize_transform, sam),
         'boxes': resize_transform.apply_boxes_torch(input_boxes[0], input_boxes[0].shape[:2]),
         'original_size': input_boxes[0].shape[:2]
     }
    ]

    batched_output = sam(batched_input, multimask_output=False)

    return batched_output

def prepare_image_embeddings(image):
    #Loading SAM predictor
    sam_predictor = LoadSAMPredictor(sam_checkpoint=SAM_CHECKPOINT,model_type=MODEL_TYPE,device='cpu')

    #Create SAM embeddings for the image
    sam_predictor.set_image(image)

    return sam_predictor

def GenerateMaskForImage(sam_predictor, bounding_boxes=[],open=False,close=False,kernel_size=None):

    masks=[]

    iou_predictions = []

    #Generate mask
    if len(bounding_boxes) == 1 :

        if type(bounding_boxes[0]) == dict:
            input_bbox = np.array(format_bbox(bounding_boxes[0]['bbox']))
        else:
            input_bbox = np.array(format_bbox(bounding_boxes[0]))

        mask, iou = GenerateMask(sam_predictor,input_box=input_bbox)
        iou_predictions.append(iou)
        if open and not(close):
            mask[0] = open_mask(np.array(mask[0]*1,dtype=np.uint8))
            masks.append(mask)
        elif close and not(open):
            mask[0] = close_mask(np.array(mask[0]*1,dtype=np.uint8))
            masks.append(mask)
        elif open and close:
            mask[0] = close_mask(close_mask(np.array(mask[0]*1,dtype=np.uint8)))
            masks.append(mask)
        else:
            masks.append(mask)
    elif len(bounding_boxes) > 1 :
        for i in range(len(bounding_boxes)):

            if type(bounding_boxes[i]) == dict:
                input_bbox = np.array(format_bbox(bounding_boxes[i]['bbox']))
            else:
                input_bbox = np.array(format_bbox(bounding_boxes[i]))

            mask, iou = GenerateMask(sam_predictor,input_box=input_bbox)
            iou_predictions.append(iou)
            if open and not(close):
                mask[0] = open_mask(np.array(mask[0]*1,dtype=np.uint8),kernel_size=kernel_size)
                masks.append(mask)
            elif close and not(open):
                mask[0] = close_mask(np.array(mask[0]*1,dtype=np.uint8),kernel_size=kernel_size)
                masks.append(mask)
            elif open and close:
                mask[0] = open_mask(close_mask(np.array(mask[0]*1,dtype=np.uint8),kernel_size=kernel_size))
                masks.append(mask)
            else:
                masks.append(mask)

    return masks,iou_predictions




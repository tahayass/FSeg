import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sam_model import LoadSAMPredictor,GenerateMask
from utils import show_box,show_mask
from postprocessing import close_mask,open_mask

from annotation_scripts.ExtractBoundingBox import load_coco_data, get_bounding_boxes_for_all_image, get_image_info

def format_bbox(input_bbox):
    #format the bounding box values to sam model bbox specifications
    return [input_bbox[0], input_bbox[1], input_bbox[0]+input_bbox[2], input_bbox[1]+input_bbox[3]]



def GenerateMasksForDataset(coco_dataset_path,dataset_name='',max_images=30,open=False,close=False):

    experiment_path = os.path.join('.','FoodAreaSegmentation','experiments',dataset_name)

    if os.path.exists(experiment_path) == False:
        os.mkdir(os.path.join('.','FoodAreaSegmentation','experiments',dataset_name))

    annotation_file_path = os.path.join(coco_dataset_path,'_annotations.coco.json')

    coco, category_id_to_name = load_coco_data(annotation_file_path)

    image_annotation = get_bounding_boxes_for_all_image(coco,category_id_to_name)
    

    image_ids = coco.getImgIds()

    for id in tqdm(image_ids[:max_images]):

        image = cv2.imread(os.path.join(coco_dataset_path,get_image_info(coco, id)['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = GenerateMaskForImage(image, bounding_boxes=image_annotation[id],open=open,close=close)

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for i,mask in enumerate(masks):
            show_mask(mask[0], plt.gca())
            category_name = image_annotation[id][i]['category_name']
            show_box(format_bbox(image_annotation[id][i]['bbox']), plt.gca(),category_name=category_name)
        plt.savefig(os.path.join(experiment_path,f'{id}.jpg'))


def GenerateMaskForImage(image, bounding_boxes=[],open=False,close=False):

    masks=[]
    #Loading SAM predictor
    sam_predictor = LoadSAMPredictor(sam_checkpoint=SAM_CHECKPOINT,model_type=MODEL_TYPE,device='cpu')

    #Create SAM embeddings for the image
    sam_predictor.set_image(image)

    #Generate mask
    if len(bounding_boxes) == 1 :
        input_bbox = np.array(format_bbox(bounding_boxes[0]['bbox']))
        mask = GenerateMask(sam_predictor,input_box=input_bbox)
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
            input_bbox = np.array(format_bbox(bounding_boxes[i]['bbox']))
            mask = GenerateMask(sam_predictor,input_box=input_bbox)
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

    return masks






if __name__=='__main__':
    SAM_CHECKPOINT = os.path.join('.','FoodAreaSegmentation','sam_vit_h_4b8939.pth')
    MODEL_TYPE = "vit_h"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    coco_dataset_path = r'.\Data\Potato Pancake.v2i.coco\test'

    GenerateMasksForDataset(coco_dataset_path,dataset_name='test new dataset open',max_images=5,close=True)
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sam_model import GenerateMaskForImage
from utils import show_box,show_mask,format_bbox

from annotation_scripts.ExtractBoundingBox import load_coco_data, get_bounding_boxes_for_all_image, get_image_info



def GenerateMasksForDataset(coco_dataset_path,dataset_name='',max_images=30,open=False,close=False,kernel_size=(10,10)):

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
        
        masks, iou_predictions = GenerateMaskForImage(image,
                                                      bounding_boxes=image_annotation[id],
                                                      open=open,
                                                      close=close,
                                                      kernel_size=kernel_size)

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for i,mask in enumerate(masks):
            show_mask(mask[0], plt.gca())
            category_name = image_annotation[id][i]['category_name']
            show_box(format_bbox(image_annotation[id][i]['bbox']),iou_predictions[i][0], plt.gca(),category_name=category_name)
        plt.savefig(os.path.join(experiment_path,f'{id}.jpg'))




if __name__=='__main__':
    SAM_CHECKPOINT = os.path.join('.','FoodAreaSegmentation','sam_vit_h_4b8939.pth')
    MODEL_TYPE = "vit_h"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    coco_dataset_path = r'.\Data\combined_dataset\test'

    GenerateMasksForDataset(coco_dataset_path,
                            dataset_name='test with boxes and points and postprocessing',
                            max_images=8,
                            open=True,
                            close=True,
                            kernel_size=(10,10)
                            )
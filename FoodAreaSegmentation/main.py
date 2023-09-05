import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sam_model import LoadSAMPredictor,GenerateMask
from utils import show_box,show_mask




if __name__=='__main__':
    SAM_CHECKPOINT = os.path.join('.','FoodAreaSegmentation','sam_vit_h_4b8939.pth')
    MODEL_TYPE = "vit_h"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    INPUT_BOX = np.array([66,219,66+520,219+317]) #get bbox from embeddinfs

    test_image = cv2.imread(r'.\Data\Fseg-food-detection-1\train\9_jpg.rf.05b03592b319ba209316ff6cf27fd6c9.jpg')
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

    #Loading SAM predictor
    print("Loading SAM predictor")
    sam_predictor = LoadSAMPredictor(sam_checkpoint=SAM_CHECKPOINT,model_type=MODEL_TYPE,device=DEVICE)

    #Create SAM embeddings for the image
    print("Creating image embedding")
    sam_predictor.set_image(test_image)

    #Generate mask
    print("Generating food mask")
    masks = GenerateMask(sam_predictor,input_box=INPUT_BOX)


    #Show result 
    category_name = 'seasoned spinach'


    plt.figure(figsize=(10, 10))
    plt.imshow(test_image)
    show_mask(masks[0], plt.gca())
    show_box(INPUT_BOX, plt.gca(),category_name=category_name)
    plt.axis('off')
    plt.show()
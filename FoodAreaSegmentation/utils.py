import matplotlib.pyplot as plt
import numpy as np



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax, iou=0, category_name=None):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    if category_name is not None:
        ax.text(x0, y0 - 5, category_name, color='green', fontsize=12, backgroundcolor='black')
    if iou is not None:
        ax.text(x0, y0 + h + 5, f'IoU: {iou:.2f}', color='red', fontsize=12, backgroundcolor='black')

def format_bbox(input_bbox):
    #format the bounding box values to sam model bbox specifications
    return [input_bbox[0]-int(input_bbox[2]/2), input_bbox[1]-int(input_bbox[3]/2), input_bbox[0]+int(input_bbox[2]/2), input_bbox[1]+int(input_bbox[3]/2)]
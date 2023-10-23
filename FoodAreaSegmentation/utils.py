import matplotlib.pyplot as plt
import numpy as np
import cv2



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

def show_mask_cv2(mask, image, random_color=False):
    if random_color:
        color = np.random.rand(3) * 255
    else:
        color = (30, 144, 255)

    mask_image = np.zeros_like(image)
    mask_image[:, :, 0] = mask * color[0]
    mask_image[:, :, 1] = mask * color[1]
    mask_image[:, :, 2] = mask * color[2]

    image = cv2.addWeighted(image, 1, mask_image, 0.2, 0)

    return image

def show_points_cv2(coords, labels, image, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    
    for point in pos_points:
        cv2.drawMarker(image, (int(point[0]), int(point[1])), (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=marker_size, thickness=2, line_type=cv2.LINE_AA)
    
    for point in neg_points:
        cv2.drawMarker(image, (int(point[0]), int(point[1])), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=marker_size, thickness=2, line_type=cv2.LINE_AA)
    
    return image

def show_box_cv2(box, image, iou=0, category_name=None):
    x0, y0, x1, y1 = map(int, box)
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
    
    if category_name is not None:
        cv2.putText(image, category_name, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if iou is not None:
        cv2.putText(image, f'IoU: {iou:.2f}', (x0, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return image
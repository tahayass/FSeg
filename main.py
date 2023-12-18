import torch
import os
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import json
from threading import Thread, Event

# Get the absolute path of the project root directory
parent_dir = os.path.dirname(os.path.realpath(__file__))
plate_detection_path = os.path.join(parent_dir,'PlateDetection')
area_segmentation_path = os.path.join(parent_dir,'FoodAreaSegmentation')

sys.path.append(parent_dir)
sys.path.append(plate_detection_path)
sys.path.append(area_segmentation_path)

# Define variables to store the return values
bboxes_result = None
embeddings_result = None

# Define events to signal when each thread has finished
bboxes_done_event = Event()
embeddings_done_event = Event()
packaged_bboxes_done_event = Event()

from FoodAreaSegmentation.sam_model import GenerateMaskForImage,prepare_image_embeddings
from FoodAreaSegmentation.utils import show_box,show_mask,format_bbox,show_box_cv2,show_mask_cv2


from PlateDetection.utils.torch_utils import select_device, smart_inference_mode
from PlateDetection.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from PlateDetection.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from PlateDetection.models.common import DetectMultiBackend
from ultralytics.utils.plotting import Annotator, colors, save_one_box
import argparse
import csv
import platform
from pathlib import Path



# Add the project root directory to the Python path if not already present
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

if project_root not in sys.path:
    sys.path.append(project_root)

def get_plate_placement(image,
                        model,
                        resize=None):
    return None


@smart_inference_mode()
def get_food_bboxes(
        weights='C:/Users/Sarah Benabdallah/Documents/GitHub/FSeg/PlateDetection/yolov5s.pt',  # model path or triton URL
        source='C:/Users/Sarah Benabdallah/Documents/GitHub/FSeg/PlateDetection/data/images',  # file/dir/URL/glob/screen/0(webcam)
        data='C:/Users/Sarah Benabdallah/Documents/GitHub/FSeg/PlateDetection/data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='C:/Users/Sarah Benabdallah/Documents/GitHub/FSeg/PlateDetection/runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / 'predictions.csv'

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        bboxes = []
        food_types = []
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    # Append bbox and food_type to lists
                    bbox = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).squeeze(0).tolist()
                    food_type = names[c]
                    bboxes.append(bbox)
                    food_types.append(food_type)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    return bboxes, food_types  # Return bboxes and food_types


@smart_inference_mode()
def get_packaged_food_bboxes(
        weights='./PlateDetection/best5food.pt', 
        source='./PlateDetection/test_images',  
        data='./PlateDetection/data/types5food.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='./FSeg/PlateDetection/runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / 'predictions.csv'

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        bboxes = []
        packaged_food_type = []
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    # Append bbox and food_type to lists
                    bbox = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).squeeze(0).tolist()
                    food_type = names[c]
                    bboxes.append(bbox)
                    packaged_food_type.append(food_type)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    return bboxes, packaged_food_type  # Return bboxes and packaged_food_type



def get_food_masks(sam_predictor,
                   bboxes,
                   open=True,
                   close=True,
                   kernel_size=None):
    
    masks,iou = GenerateMaskForImage(sam_predictor, bounding_boxes=bboxes,open=open,close=close,kernel_size=kernel_size)
    
    return masks,iou

def calculate_surface_area(masks,
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
        pixel_count[name] = non_zero_count.item()

    return pixel_count


def get_food_bboxes_worker(opt):
    global bboxes_result
    bboxes_result = get_food_bboxes(
        weights=opt["weights"],
        source=opt["source"],
        data=opt["data"],
        imgsz=opt["imgsz"],
        conf_thres=opt["conf_thres"],
        iou_thres=opt["iou_thres"],
        max_det=opt["max_det"],
        device=opt["device"],
        view_img=opt["view_img"],
        save_txt=opt["save_txt"],
        save_csv=opt["save_csv"],
        save_conf=opt["save_conf"],
        save_crop=opt["save_crop"],
        nosave=opt["nosave"],
        classes=opt["classes"],
        agnostic_nms=opt["agnostic_nms"],
        augment=opt["augment"],
        visualize=opt["visualize"],
        update=opt["update"],
        project=opt["project"],
        name=opt["name"],
        exist_ok=opt["exist_ok"],
        line_thickness=opt["line_thickness"],
        hide_labels=opt["hide_labels"],
        hide_conf=opt["hide_conf"],
        half=opt["half"],
        dnn=opt["dnn"],
        vid_stride=opt["vid_stride"]
    )
    bboxes_done_event.set()

def prepare_image_embeddings_worker(image,model_type):
    global embeddings_result
    embeddings_result = prepare_image_embeddings(image,model_type)
    embeddings_done_event.set()


def get_packaged_food_bboxes_worker(opt):
    global packaged_bboxes_result
    packaged_bboxes_result = get_packaged_food_bboxes(
        weights=opt["weights_packagedfood"],
        source=opt["source"],
        data=opt["data"],
        imgsz=opt["imgsz"],
        conf_thres=opt["conf_thres"],
        iou_thres=opt["iou_thres"],
        max_det=opt["max_det"],
        device=opt["device"],
        view_img=opt["view_img"],
        save_txt=opt["save_txt"],
        save_csv=opt["save_csv"],
        save_conf=opt["save_conf"],
        save_crop=opt["save_crop"],
        nosave=opt["nosave"],
        classes=opt["classes"],
        agnostic_nms=opt["agnostic_nms"],
        augment=opt["augment"],
        visualize=opt["visualize"],
        update=opt["update"],
        project=opt["project"],
        name=opt["name"],
        exist_ok=opt["exist_ok"],
        line_thickness=opt["line_thickness"],
        hide_labels=opt["hide_labels"],
        hide_conf=opt["hide_conf"],
        half=opt["half"],
        dnn=opt["dnn"],
        vid_stride=opt["vid_stride"]
    )
    packaged_bboxes_done_event.set()



def pipeline(opt):
    #Constant variables
    #SAM_CHECKPOINT = os.path.join('.','FoodAreaSegmentation','sam_vit_h_4b8939.pth')
    #MODEL_TYPE = "vit_h"
    #DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    image = cv2.imread(opt["source"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Create two threads to run get_food_bboxes and prepare_image_embeddings concurrently
    #get_bboxes_thread = Thread(target=get_food_bboxes_worker, args=(opt,))
    get_bboxes_thread = Thread(target=get_food_bboxes_worker,args=(opt,))
    get_packaged_bboxes_thread = Thread(target=get_packaged_food_bboxes_worker,args=(opt,))
    prepare_embeddings_thread = Thread(target=prepare_image_embeddings_worker, args=(image,opt["segmentation_model_type"]))

    # Start the threads
    get_bboxes_thread.start()
    if opt["segment"]:
        prepare_embeddings_thread.start()
    get_packaged_bboxes_thread.start()

    # Wait for both threads to finish
    get_bboxes_thread.join()
    if opt["segment"]:
        prepare_embeddings_thread.join()
    get_packaged_bboxes_thread.join()

    # Now you can access the return values
    bboxes_done_event.wait()
    if opt["segment"]:
        embeddings_done_event.wait()
    packaged_bboxes_done_event.wait()

    # Access the return values
    bboxes,food_types = bboxes_result
    sam_predictor = embeddings_result
    packaged_bboxes,packaged_food_types = packaged_bboxes_result
    
    if (len(bboxes) != 0) & opt["segment"]:
        masks,iou = get_food_masks(sam_predictor,
                            bboxes,
                            open=True,
                            close=True,
                            kernel_size=None)
    else : 
        masks = None
        iou = None
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    #Image visualization
    if masks:
        for i,mask in enumerate(masks):
            image = show_mask_cv2(mask[0],image)

    for i,bbox in enumerate(bboxes):
        if masks:
            image = show_box_cv2(format_bbox(bbox), image,iou=iou[i][0],category_name=food_types[i])
        else:
            image = show_box_cv2(format_bbox(bbox), image,iou=None,category_name=food_types[i])

    for i,bbox in enumerate(packaged_bboxes):
        image = show_box_cv2(format_bbox(bbox), image,iou=None,category_name=packaged_food_types[i])

    if opt["save"]:
        if os.path.exists(r'./PipelineTestResults') == False:
            os.mkdir(r'./PipelineTestResults')
        cv2.imwrite(os.path.join('.','PipelineTestResults',f'test.jpg'), image)

    
    #Calculates masks pixel count and returns a dictionnary with surface area for every food {'food_type':pixel_count}
    if masks :
        pixel_count_dict = calculate_surface_area(
                                    masks,
                                    food_types)
    else:
        pixel_count_dict = {}
    
    bbox_dict = {}
    packaged_bbox_dict = {}

    # Iterate over each bbox and its corresponding name
    for bbox, name in zip(bboxes, food_types):
        if name not in bbox_dict:
            bbox_dict[name] = [bbox]
        else: 
            bbox_dict[name].append(bbox)

    # Iterate over each bbox and its corresponding name
    for bbox, name in zip(packaged_bboxes, packaged_food_types):
        if name not in packaged_bbox_dict:
            packaged_bbox_dict[name] = [bbox]
        else: 
            packaged_bbox_dict[name].append(bbox)
    
    return pixel_count_dict,bbox_dict,packaged_bbox_dict,image


if __name__ == '__main__':
    start_time = time.time()
    
    # Define Arguments of Food Detection
    opt = {
        "weights": "./PlateDetection/bestnewdataset.pt",
        "weights_packagedfood" : "./PlateDetection/best5food.pt",
        "segmentation_model_type": "vit_b",
        "source": "./uploads/set1.jpg",
        "data": "./PlateDetection/data/coco128.yaml",
        "segment": True,
        "imgsz": (640, 640),
        "conf_thres": 0.25,
        "iou_thres": 0.45,
        "max_det": 1000,
        "device": '',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        "view_img": False,
        "save_txt": False,
        "save_csv": False,
        "save_conf": False,
        "save_crop": False,
        "nosave": False,
        "classes": None,  # filter by class: --class 0, or --class 0 2 3
        "agnostic_nms": False,  # class-agnostic NMS
        "augment": False,  # augmented inference
        "visualize": False,  # visualize features
        "update": False,  # update all models
        "project": "./PlateDetection/runs/detect",
        "name": "exp",
        "exist_ok": False,  # existing project/name ok, do not increment
        "line_thickness": 3,  # bounding box thickness (pixels)
        "hide_labels": False,  # hide labels
        "hide_conf": False,  # hide confidences
        "half": False,  # use FP16 half-precision inference
        "dnn": False,  # use OpenCV DNN for ONNX inference
        "vid_stride": 1,  # video frame-rate stride
        "save": True
    }

    pixel_count_dict,bbox_dict,packaged_bbox_dict,_ = pipeline(opt)

    end_time = time.time()

    print('Elapsed time : ', end_time - start_time)

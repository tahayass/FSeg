from fastapi import FastAPI, File, UploadFile
from PIL import Image
import os
import json
import base64
from main import pipeline


app = FastAPI()

# Directory to save temporary uploaded files
upload_dir = "uploads"

# Create the directory if it doesn't exist
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

def full_pipeline(input_image):
    processed_image = Image.open(input_image)
    pixel_count = {'test': 'test'}
    return processed_image, pixel_count

@app.post("/uploadfile/")
async def upload_file(file: UploadFile):

    # Define Arguments of Food Detection
    OPT = {
        "weights": "./PlateDetection/best86yolovm.pt",
        "source": os.path.join(upload_dir, file.filename),
        "data": "./PlateDetection/data/coco128.yaml",
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
        "vid_stride": 1  # video frame-rate stride
    }

    # Save the uploaded file to a temporary location
    with open(os.path.join(upload_dir, file.filename), "wb") as f:
        f.write(file.file.read())

    # Transform the image
    input_image = os.path.join(upload_dir, file.filename)
    transformed_image, pixel_count = full_pipeline(input_image)

    # Save the transformed image to a temporary location
    transformed_image_path = os.path.join(upload_dir, "transformed_" + file.filename)
    transformed_image.save(transformed_image_path)

    # Get the MIME type of the image
    extension = os.path.splitext(file.filename)[1].removeprefix('.')
    mime_type = f"image/{extension}" 

    # Encode the transformed image as Base64
    with open(transformed_image_path, "rb") as img_file:
        transformed_image_data = base64.b64encode(img_file.read()).decode("utf-8")

    # Create a response JSON that includes the Base64-encoded image, pixel_count, and MIME type
    response_data = {
        "transformed_image": transformed_image_data,
        "image_info": pipeline(OPT),
        "mime_type": mime_type
    }

    return response_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

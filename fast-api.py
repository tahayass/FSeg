from fastapi import FastAPI, File, UploadFile
from PIL import Image
import os
import json
import base64

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
        "image_info": pixel_count,
        "mime_type": mime_type
    }

    return response_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

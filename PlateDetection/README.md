# Plate Detection Using YOLO

This repository contains the code to detect plates using the YOLO model. Follow the instructions below to set up and run the model on your images.

## Setup

1. Navigate to the `PlateDetection` directory:
    ```
    cd PlateDetection
    ```

2. Install the required packages:
    ```
    pip install -r requirements.txt
    ```

## Running the Model

To detect plates in your images:

``` 
python detect.py --weights ./runs/train/weights/best.pt --source "path_to_your_test_images" 
```

Replace `path_to_your_test_images` with the path to your test images. For example:

```
python detect.py --weights ./runs/train/weights/best.pt --source ./test_images/
```

## Results

The results from the training process can be found in the `val/exp` directory.

---

Happy plate detecting!

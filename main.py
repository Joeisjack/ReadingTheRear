# Project: Reading the Rear
# Contact: amangor1@umbc.edu
# Description: This script uses YOLOv10 to detect vechicles in an image, crops the detected vechicles, then saves the images
# cropped. This is step one of preprocessing the data for a neural network that will extrapolate information from the rear of a vechicle.

import torch
import cv2
from ultralytics import YOLO
model = YOLO("yolov10n.pt")

def setup():
    # Making sure that everthing imports correctly, IDK what I'm doing here
    print(f"pyTorch version: {torch.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Ultralytics YOLO version: {YOLO._version}")

def main():
    img_path = "test_image2.png"
    image = cv2.imread(img_path)
    if image is not None:
        print("Image loaded successfully")
    else:
        print("Image does not exist")

    results = model(image, conf = 0.5)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]

            if label in ['car', 'truck', 'bus']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cropped_vechicle = image[y1:y2, x1:x2]

                cv2.imwrite("detected_vehicle_crop.png", cropped_vechicle)
                print(f"Detected {label} at coordinates: ({x1}, {y1}), ({x2}, {y2}) and saved the cropped image as 'detected_vehicle_crop.png'")
                break  # Stop after the first detected vehicle

if __name__ == "__main__":
    setup()
    main()


## Useful for later

## Cropping with cv2
# xmin, ymin, xmax, ymax = 100, 100, 200, 200
# cropped_image = image[ymin:ymax, xmin:xmax]

## Blurring license plates with cv2
# plate = cropped_image[ymin:ymax, xmin:xmax]
# blurred_plate = cv2.GaussianBlur(plate, (25, 25), 0)
# cropped_image[ymin:ymax, xmin:xmax] = blurred_plate
## Note that this might actually not be enough as gaussian blue can be reversed...

# Saving the modified image
# cv2.imwrite("modified_image.png", image)
# Project: Reading the Rear
# Contact: amangor1@umbc.edu
# Description: This script uses YOLOv10 to detect vechicles in an image, crops the detected vechicles, then saves the images
# cropped. This is step one of preprocessing the data for a neural network that will extrapolate information from the rear of a vechicle.

from unittest import result

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets

import cv2
from ultralytics import YOLO
model_car_recognition = YOLO("yolov10n.pt")
model_license_plate = YOLO("license-plate-finetune-v1n.pt")

import easyocr
reader = easyocr.Reader(['en'])

with open("states.txt", "r") as f:
    US_STATES = [line.strip().upper() for line in f]

def setup():
    # Making sure that everthing imports correctly, IDK what I'm doing here
    print(f"pyTorch version: {torch.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Ultralytics YOLO version: {YOLO._version}")

def main():
    img_path = "truck.jpg"
    try:
        image = cv2.imread(img_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        exit(1)

    results = model_car_recognition(image, conf = 0.5)
    cropped_vechicle = None

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model_car_recognition.names[class_id]

            if label in ['car', 'truck', 'bus']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cropped_vechicle = image[y1:y2, x1:x2]

                cv2.imwrite("detected_vehicle_crop.png", cropped_vechicle)
                print(f"Detected {label} at coordinates: ({x1}, {y1}), ({x2}, {y2}) and saved the cropped image as 'detected_vehicle_crop.png'")
                break  # Stop after the first detected vehicle

    # Get state and plate number
    get_plate_info(cropped_vechicle)

def get_plate_info(img):
    # Isolating the license plate from the already cropped vechicle image, then saving it
    plate_crop = None

    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model_license_plate.predict(
        source = image_rgb,
        conf = 0.5,
        save = False,
        show = False
    )
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[0])
            plate_crop = image_rgb[y1:y2, x1:x2]
            cv2.imwrite("detected_plate_crop.png", plate_crop)

    # Grayscaling for reading sake
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

    _, thresh = cv2.threshold(bfilter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    results = reader.readtext(bfilter)
    cv2.imwrite("plate_bfilter.png", bfilter)


    plate_number = "NOT FOUND"
    state_origin = "NOT FOUND"

    for (bbox, text, prob) in results:
        clean_text = text.upper().strip().replace(" ", "")

        if clean_text in US_STATES:
            state_origin = clean_text
        
        elif len(clean_text) >= 5 and any(char.isdigit() for char in clean_text):
            plate_number = clean_text

    print(f"License Plate Number: {plate_number}")
    print(f"State of Origin: {state_origin}")
    

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
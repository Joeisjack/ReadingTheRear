# Project: Reading the Rear
# Contact: amangor1@umbc.edu
# Description: This script uses YOLOv10 to detect vechicles in an image, crops the detected vechicles, then saves the images
# cropped. This is step one of preprocessing the data for a neural network that will extrapolate information from the rear of a vechicle.

from unittest import result

import json
import numpy as np

import torch
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# Object Detection
import cv2
from ultralytics import YOLO
model_car_recognition = YOLO("yolov10n.pt")
model_license_plate = YOLO("license-plate-finetune-v1n.pt")

model_state_detector = tf.keras.models.load_model('StatePlateModel.keras')
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

# Text Recognition
import easyocr
reader = easyocr.Reader(['en'])

def setup():
    # Making sure that everthing imports correctly, IDK what I'm doing here
    print(f"pyTorch version: {torch.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    # print(f"TensorFlow version: {tf.__version__}")
    print(f"Ultralytics YOLO version: {YOLO._version}")
    print(f"EasyOCR version: {easyocr.__version__}")

def main():
    img_path = "cali2.png"
    try:
        image = cv2.imread(img_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        exit(1)

    results = model_car_recognition(image, conf = 0.5)
    cropped_vechicle = None

    largest_area = 0
    best_crop = None
    best_coords = None
    best_label = None

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model_car_recognition.names[class_id]

            if label in ['car', 'truck', 'bus']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                width = x2 - x1
                height = y2 - y1
                area = width * height

                if area > largest_area:
                    largest_area = area
                    best_crop = image[y1:y2, x1:x2]
                    best_coords = (x1, y1, x2, y2)
                    best_label = label

    if best_crop is not None:
        cropped_vechicle = best_crop
        x1, y1, x2, y2 = best_coords
        cv2.imwrite("detected_vehicle_crop.png", cropped_vechicle)
        print(f"Saved detected_vehicle_crop.png — detected: {best_label}, coords: {best_coords}")
    else:
        print("No vechicle detected with confidence above threshold.")
        return

    # Get state and plate number
    get_plate_info(cropped_vechicle)

def make_square(img, size=224):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))

    square = np.zeros((size, size, 3), dtype=np.uint8)
    y_offset = (size - new_h) // 2
    x_offset = (size - new_w) // 2
    square[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return square

def get_plate_info(img):
    # Isolating the license plate from the already cropped vechicle image, then saving it
    raw_plate_crop = None
    state_detection_crop = None

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
            raw_plate_crop = image_rgb[y1:y2, x1:x2]

            state_detection_crop = cv2.resize(raw_plate_crop, (224, 128))

            cv2.imwrite("for_state_detector.png", cv2.cvtColor(state_detection_crop, cv2.COLOR_RGB2BGR))
            cv2.imwrite("for_ocr.png", raw_plate_crop)

    
    # Grayscaling for reading sake
    gray = cv2.cvtColor(raw_plate_crop, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

    _, thresh = cv2.threshold(bfilter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    results = reader.readtext(bfilter)
    cv2.imwrite("plate_bfilter.png", bfilter)


    plate_number = "NOT FOUND"
    state_origin = "NOT FOUND"
    confidence = 0.0

    # Getting state of origin
    state_origin, confidence = predict_plate_state(state_detection_crop)

    for (bbox, text, prob) in results:
        clean_text = text.upper().strip().replace(" ", "")
        
        if (len(clean_text) >= 5 and any(char.isdigit() for char in clean_text)):
            plate_number = clean_text

    print(f"License Plate Number: {plate_number}")
    print(f"State of Origin: {state_origin}, Confidence: {confidence:.2f}")

def predict_plate_state(plate_crop):
    img_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (320, 160))

    img_rgb = make_square(img_rgb, size = 224)

    img_processed = preprocess_input(img_rgb.astype(np.float32))

    input_tensor = np.expand_dims(img_processed, axis = 0)

    

    cv2.imwrite("eeee.png", img_rgb)

    predictions = model_state_detector.predict(input_tensor, verbose = 0)
    
    preds = predictions[0]

    top5_indices = np.argsort(preds)[-5:][::-1]

    print("Top 5 State Predictions:")
    for i in top5_indices:
        print(f"{labels[i]}: {preds[i]:.4f}")

    class_idx = top5_indices[0]
    score = preds[class_idx]

    return labels[class_idx], score

    

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
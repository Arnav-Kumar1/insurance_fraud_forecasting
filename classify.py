import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import random

def process_claim(a,b):
    return random.randint(a,b)

def import_rf_model():
    damage_list = ['minor', 'moderate', 'severe']
    damage = damage_list[random.randint(0,2)]
    return damage

# Run the app
if __name__ == '__main__':

    model = YOLO('damage_classifier.pt')

    image_path = "Damage_classification_dataset/test/minor/0094.JPEG"

    # Perform object classification

    result = model(image_path)

    names_dict = result[0].names

    probs = result[0].probs

    label = names_dict[probs.top1]


    print(label)

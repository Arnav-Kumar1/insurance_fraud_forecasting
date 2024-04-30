import cv2
from ultralytics import YOLO
import supervision as sv

# Mapping dictionary for class IDs to labels
class_mapping = {
    0: "Bonnet",
    1: "Bumper",
    2: "Dickey",
    3: "Door",
    4: "Fender",
    5: "Light",
    6: "Windshield"
}

def main():
    image_path = "car_damage_detection_split/train/images/0001_JPEG.rf.8cce9bb7a46ff7494b475d1ab652324a.jpg"
    model = YOLO("models/best.pt")

    frame = cv2.imread(image_path)

    box_annotator = sv.BoxAnnotator(
        thickness=1,
        text_thickness=2,
        text_scale=1
    )

    result = model(frame, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)

    labels = []

    # Create labels for all detected objects using the mapping dictionary
    for detection in detections:
        class_id = detection[2]  # Assuming class ID is at index 2
        confidence = detection[1]  # Assuming confidence is at index 1
        label = f"{class_mapping.get(class_id, 'Unknown')} {confidence:.2f}"
        labels.append(label)

    # Annotate the frame with detections and labels
    frame = box_annotator.annotate(
        scene=frame,
        detections=detections,
        labels=labels
    )

    cv2.imshow("yolov8", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # Close windows properly


if __name__ == "__main__":
    main()

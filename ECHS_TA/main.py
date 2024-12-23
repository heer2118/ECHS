import cv2
import numpy as np
import math
import time


# Load YOLO Model
def load_yolo():
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    layer_names = net.getLayerNames()

    # Handle the output layer extraction
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, output_layers


# Detect players using YOLO
def detect_players(frame, net, output_layers):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Only keep "person" class (YOLO class id for person is 0)
            if confidence > 0.5 and class_id == 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    players = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            players.append((x, y, w, h))

    return players


# Calculate speed using position differences
def calculate_speed(old_positions, new_positions, fps):
    speeds = []
    for old_pos, new_pos in zip(old_positions, new_positions):
        if old_pos is not None and new_pos is not None:
            x1, y1, _, _ = old_pos
            x2, y2, _, _ = new_pos
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            speed = distance * fps  # pixel/frame to pixel/sec
            speeds.append(speed)
    return speeds


# Estimate energy level based on speed
def estimate_energy_level(speeds):
    energy_levels = []
    for speed in speeds:
        if speed > 10:  # Adjust threshold for high speed
            energy_levels.append("High")
        elif 5 < speed <= 10:
            energy_levels.append("Medium")
        else:
            energy_levels.append("Low")
    return energy_levels


def main():
    net, output_layers = load_yolo()
    cap = cv2.VideoCapture("football_match.mp4")  # Replace with your video source

    fps = cap.get(cv2.CAP_PROP_FPS)
    old_positions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        players = detect_players(frame, net, output_layers)
        new_positions = [(x, y, w, h) for (x, y, w, h) in players]

        if len(old_positions) == len(new_positions) and len(new_positions) > 0:
            speeds = calculate_speed(old_positions, new_positions, fps)
            energy_levels = estimate_energy_level(speeds)

            for i, (x, y, w, h) in enumerate(new_positions):
                if i < len(energy_levels):
                    color = (0, 255, 0) if energy_levels[i] == "High" else (0, 255, 255) if energy_levels[
                                                                                                i] == "Medium" else (
                    0, 0, 255)
                    label = f"{energy_levels[i]} Energy"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Update old positions if players are detected
        old_positions = new_positions if len(new_positions) > 0 else old_positions

        cv2.imshow("Football Analytics", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

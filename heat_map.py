import cv2
import numpy as np
import math

# Load YOLO Model
def load_yolo():
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, output_layers

# Detect players using YOLO
def detect_players(frame, net, output_layers):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 0:  # Only for "person" class
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

# Update heat map based on player body positions
def update_heat_map(heat_map, players):
    for (x, y, w, h) in players:
        # Draw the player's bounding box on the heat map
        heat_map[y:y+h, x:x+w] = np.clip(heat_map[y:y+h, x:x+w] + 1, 0, 255)  # Increment intensity

# Generate a color heat map
def generate_color_heat_map(heat_map):
    # Normalize the heat map to range 0-255
    heat_map_normalized = cv2.normalize(heat_map, None, 0, 255, cv2.NORM_MINMAX)
    heat_map_color = cv2.applyColorMap(heat_map_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    return heat_map_color

# Estimate energy level based on heat map intensity
def estimate_energy_level(heat_map):
    mean_intensity = np.mean(heat_map)
    if mean_intensity > 200:  # Adjust threshold as necessary
        return "High Energy"
    elif 100 < mean_intensity <= 200:
        return "Medium Energy"
    else:
        return "Low Energy"

def main():
    net, output_layers = load_yolo()
    cap = cv2.VideoCapture("football_match.mp4")  # Replace with your video source

    heat_map = np.zeros((720, 1280), np.uint8)  # Adjust size to match your video resolution

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        players = detect_players(frame, net, output_layers)
        update_heat_map(heat_map, players)

        # Generate color heat map
        heat_map_color = generate_color_heat_map(heat_map)

        # Estimate energy level
        energy_level = estimate_energy_level(heat_map)

        # Display results
        cv2.putText(frame, f"Energy Level: {energy_level}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Football Analytics", frame)
        cv2.imshow("Heat Map", heat_map_color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

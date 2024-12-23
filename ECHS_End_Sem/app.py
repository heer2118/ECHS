import os
import cv2
import torch
import numpy as np
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, render_template, send_from_directory, flash
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")



app = Flask(__name__)

# Secret key for session management (needed for flash messages)
app.secret_key = os.urandom(24)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load the YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Video Processing Logic
def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    player_positions = defaultdict(list)
    ball_position = None
    ball_positions = []
    ball_trail = []
    scale = 0.05
    speed_map = {}
    speed_display_duration = 2 * fps
    
    def categorize_speed(speed):
        if speed == 0:
            return "Idle", (50, 205, 154)
        elif 0 < speed <= 3:
            return "Walking", (0, 255, 0)
        elif 4 <= speed < 9:
            return "Running", (0, 255, 255)
        else:
            return "Sprinting", (0, 140, 255)
    
    font_path = "path/to/verdana.ttf"  # Make sure to replace with the correct path
    font = ImageFont.truetype(font_path, 18)
    
    frame_count = 0
    frame_speed_map = defaultdict(int)
    frame_speed_timestamp = defaultdict(int)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)
        detections = results.xyxy[0].cpu().numpy()

        person_detections = [d for d in detections if int(d[5]) == 0]
        ball_detections = [d for d in detections if int(d[5]) == 32]

        if len(ball_detections) > 0:
            x1, y1, x2, y2, conf = ball_detections[0][:5]
            ball_position = (int((x1 + x2) // 2), int((y1 + y2) // 2))
            ball_positions.append(ball_position)

            ball_trail.append((ball_position, frame_count))
            ball_trail = [pos for pos in ball_trail if frame_count - pos[1] <= 60]

            for i, (position, trail_frame) in enumerate(ball_trail):
                if frame_count - trail_frame <= 30:
                    opacity = 255 - int(255 * (frame_count - trail_frame) / 30)
                    trail_radius = 3
                    cv2.circle(frame, position, trail_radius, (0, 0, 0, opacity), -1)

        for idx, det in enumerate(person_detections):
            x1, y1, x2, y2, conf = det[:5]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if idx not in player_positions:
                player_positions[idx] = [(center_x, center_y, frame_count)]
            else:
                player_positions[idx].append((center_x, center_y, frame_count))

            speed = 0.0
            if len(player_positions[idx]) >= 2:
                x1_prev, y1_prev, t1 = player_positions[idx][-2]
                x2_curr, y2_curr, t2 = player_positions[idx][-1]
                distance = np.sqrt((x2_curr - x1_prev) ** 2 + (y2_curr - y1_prev) ** 2) * scale
                time_diff = (t2 - t1) / fps
                if time_diff > 0:
                    speed = distance / time_diff

                if speed > 10:
                    speed = 10
                elif speed < 0:
                    speed = 0
                speed_map[idx] = speed
                frame_speed_timestamp[idx] = frame_count

            if idx in speed_map and (frame_count - frame_speed_timestamp.get(idx, -speed_display_duration)) >= speed_display_duration:
                speed = speed_map[idx]

            speed_category, box_color_bgr = categorize_speed(speed)
            box_color_rgb = (box_color_bgr[2], box_color_bgr[1], box_color_bgr[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color_bgr, 2)

            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            if (frame_count - frame_speed_timestamp.get(idx, -speed_display_duration)) <= speed_display_duration:
                speed_label = f"{speed:.2f} m/s"
                speed_category_label = speed_category

                label_padding = 10
                draw.text((x1, y1 - 50), speed_label, font=font, fill=box_color_rgb)
                draw.text((x1, y1 - 30), speed_category_label, font=font, fill=box_color_rgb)

            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

@app.route('/')
def index():
    # flash("Welcome to the video processing site!")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file', 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_video_path = os.path.join(app.config['OUTPUT_FOLDER'], f"processed_{filename}")

        # Save the uploaded file
        file.save(input_video_path)

        # Flash message for processing start
        # flash("Video is being processed. Please wait...")

        # Process the video
        process_video(input_video_path, output_video_path)

        # Flash message for processing completion
        flash("Video processing is complete! You can now download the processed video.")

        # Return the processed video
        return send_from_directory(app.config['OUTPUT_FOLDER'], f"processed_{filename}", as_attachment=True)

    return 'Invalid file type', 400

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    app.run(debug=True)

import cv2
import torch
import time
import numpy as np
from flask import Flask, Response, render_template, request, jsonify

app = Flask(__name__)

# Muat model YOLOv5 (menggunakan pretrained model 'yolov5s')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
TARGET_CLASSES = ['car', 'motorcycle', 'truck', 'bus', 'person']
classnames = model.names

# URL RTSP kamera
rtsp_url = "rtsp://olean:cctvmadiun123@10.10.122.33:554/streaming/channels/2"

# Variabel global untuk menghitung objek
count_dict = {'car': 0, 'motorcycle': 0, 'truck': 0, 'bus': 0, 'person': 0}
tracked_objects = {}

# --- Global variable untuk koordinat garis ---
# Default garis (misalnya, dimulai di (50,50) dan berakhir di (250,250))
line_coords = {"x1": 50, "y1": 50, "x2": 250, "y2": 250}

def generate_frames():
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka stream RTSP")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame tidak diterima")
            break

        height, width, _ = frame.shape

        # Lakukan object detection menggunakan YOLOv5
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()
        for *box, conf, cls in detections:
            class_name = classnames[int(cls)]
            if class_name in TARGET_CLASSES:
                x1, y1, x2, y2 = map(int, box)
                centroid_x = (x1 + x2) // 2
                centroid_y = (y1 + y2) // 2

                # Perhitungan sederhana: misalnya, kita anggap objek melewati garis
                # jika centroid berada "di bawah" garis. Kita hitung persamaan garis:
                # y = m*x + c, di mana m = (y2 - y1)/(x2 - x1) dan c = y1 - m*x1.
                # Gunakan koordinat dari line_coords.
                m = (line_coords["y2"] - line_coords["y1"]) / (line_coords["x2"] - line_coords["x1"] + 1e-5)
                c = line_coords["y1"] - m * line_coords["x1"]
                line_y_at_centroid = m * centroid_x + c

                if centroid_y > line_y_at_centroid:
                    obj_id = f"{class_name}_{centroid_x}_{centroid_y}"
                    if obj_id not in tracked_objects:
                        tracked_objects[obj_id] = True
                        count_dict[class_name] += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Gambar garis deteksi menggunakan koordinat yang dapat diupdate
        cv2.line(frame, (line_coords["x1"], line_coords["y1"]),
                      (line_coords["x2"], line_coords["y2"]), (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

# Endpoint untuk mengupdate koordinat garis secara interaktif via POST
@app.route('/set_line', methods=['POST'])
def set_line():
    global line_coords
    data = request.get_json()
    try:
        line_coords["x1"] = int(data.get("x1", line_coords["x1"]))
        line_coords["y1"] = int(data.get("y1", line_coords["y1"]))
        line_coords["x2"] = int(data.get("x2", line_coords["x2"]))
        line_coords["y2"] = int(data.get("y2", line_coords["y2"]))
        return jsonify({"status": "success", "line_coords": line_coords})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/get_counts')
def get_counts():
    return jsonify(count_dict)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

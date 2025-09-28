import os
import cv2
import time
import math
import numpy as np
import threading, queue, sqlite3, pathlib
import pandas as pd
from io import BytesIO
import asyncio
import json

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO  # Menggunakan YOLOv8

# Agar kompatibel dengan WindowsPath (jika diperlukan)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = FastAPI()

# Tentukan BASE_DIR untuk path absolut
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ---------------------------------------------------------------
# Konfigurasi Kamera, Model, dan Variabel Global
# ---------------------------------------------------------------
CAMERAS = {
    "psc": {
        "name": "JEMBATAN PENYEBERANGAN PSC",
        "rtsp": "rtsp://olean:cctvmadiun123@10.10.122.33:554/streaming/channels/2"
    },
    "trotoar": {
        "name": "JALAN URIP SUMOHARJO",
        "rtsp": "rtsp://admin:dishub2024@10.10.77.130:554/cam/realmonitor?chh"
    },
    "tugu": {
        "name": "TUGU 0 KM MADIUN",
        "rtsp": "rtsp://admin:kominfo123@10.10.122.62:554"
    },
    "gajah": {
        "name": "JALAN GAJAH MADA ARAH SELATAN",
        "rtsp": "rtsp://admin:kominfo123@10.10.122.78:554"
    }
}

TARGET_CLASSES = ['car', 'motorcycle', 'truck', 'bus', 'person', 'bicycle']

available_models = {
    "yolov8s": os.path.join(BASE_DIR, "yolov8s.pt"),
    "yolov8x": os.path.join(BASE_DIR, "yolov8x.pt"),
    "yolov8n": os.path.join(BASE_DIR, "yolov8n.pt")
}

current_model_name = "yolov8s"
current_model = YOLO(available_models[current_model_name])

tracked_objects_per_camera = {}
polygons = {}
counts_per_camera = {}
latest_frames = {}

detection_threads = {}

# ---------------------------------------------------------------
# Database Setup
# ---------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(os.path.join(BASE_DIR, 'detection_counts.db'))
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS detection_counts (
            date TEXT,
            camera_id TEXT,
            car INTEGER,
            motorcycle INTEGER,
            truck INTEGER,
            bus INTEGER,
            person INTEGER,
            bicycle INTEGER,
            PRIMARY KEY (date, camera_id)
        )
    ''')
    conn.commit()
    conn.close()

def store_daily_counts():
    today = time.strftime("%Y-%m-%d")
    conn = sqlite3.connect(os.path.join(BASE_DIR, 'detection_counts.db'))
    c = conn.cursor()
    for cam_id, count_dict in counts_per_camera.items():
        c.execute("SELECT * FROM detection_counts WHERE date=? AND camera_id=?", (today, cam_id))
        row = c.fetchone()
        if row:
            new_counts = (
                count_dict.get('car', 0) + row[2],
                count_dict.get('motorcycle', 0) + row[3],
                count_dict.get('truck', 0) + row[4],
                count_dict.get('bus', 0) + row[5],
                count_dict.get('person', 0) + row[6],
                count_dict.get('bicycle', 0) + row[7],
                today, cam_id
            )
            c.execute('''
                UPDATE detection_counts
                SET car=?, motorcycle=?, truck=?, bus=?, person=?, bicycle=?
                WHERE date=? AND camera_id=?
            ''', new_counts)
        else:
            c.execute('''
                INSERT INTO detection_counts (date, camera_id, car, motorcycle, truck, bus, person, bicycle)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                today, cam_id,
                count_dict.get('car', 0),
                count_dict.get('motorcycle', 0),
                count_dict.get('truck', 0),
                count_dict.get('bus', 0),
                count_dict.get('person', 0),
                count_dict.get('bicycle', 0)
            ))
    conn.commit()
    conn.close()

def periodic_db_store():
    while True:
        time.sleep(60)
        store_daily_counts()
        print("Daily counts stored.")

init_db()
db_thread = threading.Thread(target=periodic_db_store, daemon=True)
db_thread.start()

# ---------------------------------------------------------------
# Video Capture & Detection
# ---------------------------------------------------------------
class VideoCaptureThread:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.frame_queue = queue.Queue(maxsize=2)
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()
    
    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                print("Gagal mengambil frame dari RTSP:", self.cap)
                self.stopped = True
                break
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
        self.cap.release()
    
    def read(self):
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None
    
    def stop(self):
        self.stopped = True

def distance(pt1, pt2):
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def detection_loop(cam_id, rtsp_link):
    if cam_id not in counts_per_camera:
        counts_per_camera[cam_id] = {cls: 0 for cls in TARGET_CLASSES}
    if cam_id not in tracked_objects_per_camera:
        tracked_objects_per_camera[cam_id] = []
    
    cap_thread = VideoCaptureThread(rtsp_link)
    while True:
        frame = cap_thread.read()
        if frame is None:
            time.sleep(0.03)
            continue

        # Proses polygon (jika ada)
        poly_data = polygons.get(cam_id, None)
        pts = None
        if poly_data and "points" in poly_data and poly_data.get("canvasWidth") and poly_data.get("canvasHeight"):
            points = poly_data["points"]
            canvas_w = float(poly_data["canvasWidth"])
            canvas_h = float(poly_data["canvasHeight"])
            frame_h, frame_w = frame.shape[:2]
            scale_x = frame_w / canvas_w
            scale_y = frame_h / canvas_h
            scaled_points = [(int(pt[0] * scale_x), int(pt[1] * scale_y)) for pt in points]
            pts = np.array(scaled_points, np.int32).reshape((-1, 1, 2))
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color=(0, 0, 255))
            alpha = 0.20
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        current_time = time.time()
        tracked_objects = [obj for obj in tracked_objects_per_camera[cam_id] if current_time - obj['last_seen'] < 2.0]
        tracked_objects_per_camera[cam_id] = tracked_objects

        results = current_model(frame)
        detections = []
        if results and len(results) > 0:
            for box in results[0].boxes:
                xyxy = box.xyxy.cpu().numpy()[0]
                conf = float(box.conf.cpu().numpy()[0])
                cls = int(box.cls.cpu().numpy()[0])
                detections.append([*xyxy, conf, cls])
        
        threshold = 50
        for *box, conf, cls in detections:
            if conf < 0.40:
                continue
            class_id = int(cls)
            class_name = current_model.names[class_id] if hasattr(current_model, 'names') else str(class_id)
            if class_name not in TARGET_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box)
            current_box = (x1, y1, x2, y2)
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            computed_inside = False
            if pts is not None:
                computed_inside = cv2.pointPolygonTest(pts, centroid, False) >= 0

            matched = False
            for obj in tracked_objects:
                existing_box = obj.get('bbox', None)
                iou = compute_iou(existing_box, current_box) if existing_box else 0
                if (iou > 0.5) or (obj['class'] == class_name and distance(obj['centroid'], centroid) < threshold):
                    matched = True
                    prev_inside = obj['inside']
                    obj['centroid'] = centroid
                    obj['bbox'] = current_box
                    obj['last_seen'] = current_time
                    obj['inside'] = computed_inside
                    if not obj['counted'] and (not prev_inside) and computed_inside:
                        counts_per_camera[cam_id][class_name] += 1
                        obj['counted'] = True
                    break
            if not matched:
                new_obj = {
                    'class': class_name,
                    'centroid': centroid,
                    'bbox': current_box,
                    'inside': computed_inside,
                    'counted': computed_inside,
                    'last_seen': current_time
                }
                if computed_inside:
                    counts_per_camera[cam_id][class_name] += 1
                tracked_objects.append(new_obj)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if ret:
            latest_frames[cam_id] = buffer.tobytes()
        time.sleep(0.03)
    # Catatan: cap_thread.stop() tidak dipanggil karena loop berjalan terus menerus.

# ---------------------------------------------------------------
# Endpoint & Routing
# ---------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "cameras": CAMERAS})

@app.get("/camera/{cam_id}", response_class=HTMLResponse)
async def camera_view(request: Request, cam_id: str):
    if cam_id not in CAMERAS:
        return HTMLResponse("Camera not found", status_code=404)
    camera = CAMERAS[cam_id]
    if cam_id not in detection_threads:
        rtsp_link = camera['rtsp']
        thread = threading.Thread(target=detection_loop, args=(cam_id, rtsp_link), daemon=True)
        detection_threads[cam_id] = thread
        thread.start()
    return templates.TemplateResponse("camera.html", {"request": request, "camera": camera, "cam_id": cam_id})

@app.get("/data", response_class=HTMLResponse)
async def data_view(request: Request):
    conn = sqlite3.connect(os.path.join(BASE_DIR, 'detection_counts.db'))
    c = conn.cursor()
    c.execute("SELECT * FROM detection_counts")
    all_rows = c.fetchall()
    conn.close()
    return templates.TemplateResponse("data.html", {
        "request": request,
        "cameras": CAMERAS,
        "records": all_rows
    })

@app.get("/all", response_class=HTMLResponse)
async def all_view(request: Request):
    for cam_id, camera in CAMERAS.items():
        if cam_id not in detection_threads:
            rtsp_link = camera['rtsp']
            thread = threading.Thread(target=detection_loop, args=(cam_id, rtsp_link), daemon=True)
            detection_threads[cam_id] = thread
            thread.start()
    conn = sqlite3.connect(os.path.join(BASE_DIR, 'detection_counts.db'))
    c = conn.cursor()
    c.execute("SELECT * FROM detection_counts")
    all_rows = c.fetchall()
    conn.close()
    return templates.TemplateResponse("all.html", {
        "request": request, 
        "cameras": CAMERAS, 
        "records": all_rows,
        "current_model_name": current_model_name
    })

@app.post("/set_polygon/{cam_id}")
async def set_polygon(cam_id: str, request: Request):
    data = await request.json()
    try:
        pts = data.get("points", [])
        canvasWidth = data.get("canvasWidth")
        canvasHeight = data.get("canvasHeight")
        unique_pts = [(int(pt[0]), int(pt[1])) for pt in pts]
        polygons[cam_id] = {"points": unique_pts, "canvasWidth": canvasWidth, "canvasHeight": canvasHeight}
        counts_per_camera[cam_id] = {cls: 0 for cls in TARGET_CLASSES}
        return {"status": "success", "polygon": polygons[cam_id]}
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

@app.get("/get_polygon/{cam_id}")
async def get_polygon_endpoint(cam_id: str):
    poly = polygons.get(cam_id, {})
    return {"polygon": poly}

@app.post("/delete_polygon/{cam_id}")
async def delete_polygon(cam_id: str):
    if cam_id in polygons:
        polygons.pop(cam_id)
    return {"status": "success", "message": "Polygon deleted"}

@app.get("/get_counts/{cam_id}")
async def get_counts(cam_id: str):
    if cam_id not in counts_per_camera:
        return {}
    counts = counts_per_camera[cam_id]
    total = sum(counts.values())
    percentages = {k: (v / total * 100 if total > 0 else 0) for k, v in counts.items()}
    return {"counts": counts, "percentages": percentages}

@app.get("/export_excel")
async def export_excel():
    conn = sqlite3.connect(os.path.join(BASE_DIR, 'detection_counts.db'))
    df = pd.read_sql_query("SELECT * FROM detection_counts", conn)
    conn.close()
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='DetectionCounts')
    output.seek(0)
    return StreamingResponse(output,
                             media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             headers={"Content-Disposition": "attachment; filename=detection_counts.xlsx"})

@app.get("/export_json")
async def export_json():
    conn = sqlite3.connect(os.path.join(BASE_DIR, 'detection_counts.db'))
    c = conn.cursor()
    c.execute("SELECT * FROM detection_counts")
    rows = c.fetchall()
    conn.close()
    
    # Konversi data ke format JSON yang mudah dibaca
    data = []
    for row in rows:
        record = {
            "date": row[0],
            "camera_id": row[1],
            "camera_name": CAMERAS.get(row[1], {}).get("name", "Unknown Camera"),
            "counts": {
                "car": row[2],
                "motorcycle": row[3],
                "truck": row[4],
                "bus": row[5],
                "person": row[6],
                "bicycle": row[7]
            },
            "total": sum(row[2:8])
        }
        data.append(record)
    
    json_data = json.dumps(data, indent=2, ensure_ascii=False)
    
    return StreamingResponse(
        iter([json_data.encode('utf-8')]),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=detection_counts.json"}
    )

@app.post("/reset_data")
async def reset_data():
    conn = sqlite3.connect(os.path.join(BASE_DIR, 'detection_counts.db'))
    c = conn.cursor()
    c.execute("DELETE FROM detection_counts")
    conn.commit()
    conn.close()
    global counts_per_camera
    counts_per_camera = {cam_id: {cls: 0 for cls in TARGET_CLASSES} for cam_id in CAMERAS}
    return {"status": "success", "message": "Semua data telah direset."}

@app.post("/switch_model")
async def switch_model(request: Request):
    data = await request.json()
    model = data.get("model")
    global current_model, current_model_name
    if model not in available_models:
        return JSONResponse(status_code=400, content={"message": "Model tidak tersedia.", "status": "failed"})
    try:
        new_model = YOLO(available_models[model])
        if not hasattr(new_model, 'names'):
            raise Exception("Model tidak memiliki atribut 'names'.")
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Gagal mengganti model: {str(e)}", "status": "failed"})
    current_model = new_model
    current_model_name = model
    return {"message": f"Model berhasil diganti ke {model}.", "status": "success"}

@app.post("/next_camera")
async def next_camera(request: Request):
    data = await request.json()
    current = data.get("current")
    keys = list(CAMERAS.keys())
    try:
        idx = keys.index(current)
        next_idx = (idx + 1) % len(keys)
        next_id = keys[next_idx]
    except ValueError:
        next_id = keys[0]
    return {"id": next_id}

@app.post("/prev_camera")
async def prev_camera(request: Request):
    data = await request.json()
    current = data.get("current")
    keys = list(CAMERAS.keys())
    try:
        idx = keys.index(current)
        prev_idx = (idx - 1) % len(keys)
        prev_id = keys[prev_idx]
    except ValueError:
        prev_id = keys[0]
    return {"id": prev_id}

@app.get("/video_feed/{cam_id}")
def video_feed(cam_id: str):
    def frame_generator():
        while True:
            frame_bytes = latest_frames.get(cam_id, None)
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.05)
    return StreamingResponse(frame_generator(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.websocket("/ws/video_feed/{cam_id}")
async def websocket_video_feed(websocket: WebSocket, cam_id: str):
    await websocket.accept()
    try:
        while True:
            if cam_id in latest_frames:
                await websocket.send_bytes(latest_frames[cam_id])
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for camera {cam_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
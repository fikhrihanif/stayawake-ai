# detector.py
# Logic deteksi kantuk menggunakan MediaPipe FaceLandmarker API baru

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as mpImage
from scipy.spatial import distance
import urllib.request
import os
import time

# ─────────────────────────────────────────
# INDEX LANDMARK MATA (MediaPipe 478 points)
# ─────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ─────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────
EAR_THRESHOLD   = 0.25
FRAME_THRESHOLD = 15
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"

# ─────────────────────────────────────────
# DOWNLOAD MODEL JIKA BELUM ADA
# ─────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print("[INFO] Mendownload model MediaPipe...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[INFO] Model berhasil didownload!")

# ─────────────────────────────────────────
# FUNGSI EAR
# ─────────────────────────────────────────
def eye_aspect_ratio(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# ─────────────────────────────────────────
# INISIALISASI MEDIAPIPE (API BARU)
# ─────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

# ─────────────────────────────────────────
# FUNGSI PROSES GAMBAR
# ─────────────────────────────────────────
def process_image(frame):
    start_time = time.time()

    h, w      = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mpImage(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = face_landmarker.detect(mp_image)

    status    = "Wajah tidak terdeteksi"
    ear_value = 0.0
    is_drowsy = False
    color     = (0, 165, 255)
    output    = frame.copy()

    if result.face_landmarks:
        for face in result.face_landmarks:
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face]

            left_eye  = [landmarks[i] for i in LEFT_EYE]
            right_eye = [landmarks[i] for i in RIGHT_EYE]

            left_ear  = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear       = (left_ear + right_ear) / 2.0

            # Gambar kontur mata
            cv2.polylines(output, [np.array(left_eye)],  True, (0, 255, 0), 1)
            cv2.polylines(output, [np.array(right_eye)], True, (0, 255, 0), 1)
            for point in left_eye + right_eye:
                cv2.circle(output, point, 2, (0, 255, 0), -1)

            # Klasifikasi
            if ear < EAR_THRESHOLD:
                status    = "MENGANTUK"
                is_drowsy = True
                color     = (0, 0, 255)
            else:
                status    = "NORMAL"
                is_drowsy = False
                color     = (0, 255, 0)

            ear_value = round(ear, 4)

            cv2.putText(output, f"Status : {status}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(output, f"EAR    : {ear_value}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    inf_time = round((time.time() - start_time) * 1000, 2)

    return output, status, ear_value, is_drowsy, inf_time


# ─────────────────────────────────────────
# FUNGSI PROSES VIDEO
# ─────────────────────────────────────────
def process_video(video_path):
    cap          = cv2.VideoCapture(video_path)
    total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    frame_count  = 0
    drowsy_count = 0
    counter      = 0
    ear_list     = []
    sample_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame  = cv2.resize(frame, (640, 480))
        output, status, ear, is_drowsy, _ = process_image(frame)

        if ear > 0:
            ear_list.append(ear)

        if is_drowsy:
            counter += 1
            if counter >= FRAME_THRESHOLD:
                drowsy_count += 1
        else:
            counter = 0

        if frame_count % max(1, total // 6) == 0:
            sample_frames.append(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

    cap.release()

    drowsy_pct = round((drowsy_count / frame_count * 100), 1) if frame_count > 0 else 0
    avg_ear    = round(sum(ear_list) / len(ear_list), 4) if ear_list else 0
    duration   = round(frame_count / fps, 1) if fps > 0 else 0

    stats = {
        "total_frame"  : frame_count,
        "drowsy_frame" : drowsy_count,
        "drowsy_pct"   : drowsy_pct,
        "avg_ear"      : avg_ear,
        "duration"     : duration,
        "fps"          : round(fps, 1)
    }

    return sample_frames, stats
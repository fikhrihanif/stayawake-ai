# live.py
# Live Drowsiness Detection - Optimized

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as mpImage
from scipy.spatial import distance
import winsound
import time
import os
import threading
import csv
from datetime import datetime


# ─────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────
EAR_THRESHOLD   = 0.25
FRAME_THRESHOLD = 15
SKIP_FRAME      = 2
MODEL_PATH      = "face_landmarker.task"

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ─────────────────────────────────────────
# FUNGSI EAR
# ─────────────────────────────────────────
def eye_aspect_ratio(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

def get_confidence(ear):
    if ear == 0:
        return 0.0
    if ear < EAR_THRESHOLD:
        conf = min(1.0, (EAR_THRESHOLD - ear) / EAR_THRESHOLD * 2)
    else:
        conf = min(1.0, (ear - EAR_THRESHOLD) / EAR_THRESHOLD * 2)
    return round(conf, 4)

# ─────────────────────────────────────────
# FUNGSI ALARM (thread terpisah agar tidak block frame)
# ─────────────────────────────────────────
def play_alarm():
    winsound.Beep(1000, 1000)

# ─────────────────────────────────────────
# INISIALISASI MEDIAPIPE
# ─────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print("[ERROR] Model face_landmarker.task tidak ditemukan!")
    print("[INFO] Jalankan app.py terlebih dahulu untuk download model.")
    exit()

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options      = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

# ─────────────────────────────────────────
# INISIALISASI KAMERA
# ─────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

time.sleep(1)

if not cap.isOpened():
    print("[ERROR] Kamera tidak dapat dibuka!")
    exit()

for _ in range(5):
    cap.read()

print("[INFO] Live Drowsiness Detection dimulai...")
print("[INFO] Tekan 'Q' untuk keluar.")

# ─────────────────────────────────────────
# VARIABEL STATE
# ─────────────────────────────────────────
counter        = 0
is_drowsy      = False
total_frame    = 0
drowsy_frame   = 0
fps_list       = []
ear_value      = 0.0
status         = "Mendeteksi..."
color          = (200, 200, 200)
last_result    = None
alarm_playing  = False

# ─────────────────────────────────────────
# SETUP LOG FILE
# ─────────────────────────────────────────
LOG_FILE = "session_log.csv"
with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "status", "ear", "confidence"])

# ─────────────────────────────────────────
# LOOP UTAMA
# ─────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    total_frame += 1
    t_start      = time.time()
    frame        = cv2.flip(frame, 1)
    h, w         = frame.shape[:2]

    # ── Deteksi hanya setiap SKIP_FRAME ──
    if total_frame % SKIP_FRAME == 0:
        rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image   = mpImage(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        last_result = face_landmarker.detect(mp_image)

    # ── Proses hasil deteksi ──
    if last_result and last_result.face_landmarks:
        for face in last_result.face_landmarks:
            landmarks  = [(int(lm.x * w), int(lm.y * h)) for lm in face]
            left_eye   = [landmarks[i] for i in LEFT_EYE]
            right_eye  = [landmarks[i] for i in RIGHT_EYE]

            left_ear   = eye_aspect_ratio(left_eye)
            right_ear  = eye_aspect_ratio(right_eye)
            ear        = (left_ear + right_ear) / 2.0
            ear_value  = round(ear, 4)

            # Kontur mata
            cv2.polylines(frame, [np.array(left_eye)],  True, (0, 255, 120), 1)
            cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 120), 1)
            for point in left_eye + right_eye:
                cv2.circle(frame, point, 2, (0, 255, 120), -1)

            # Klasifikasi
            if ear < EAR_THRESHOLD:
                counter += 1
                if counter >= FRAME_THRESHOLD:
                    is_drowsy    = True
                    drowsy_frame += 1
                    

                    # Alarm di thread terpisah agar tidak block
                    if not alarm_playing:
                        alarm_playing = True
                        t = threading.Thread(target=play_alarm, daemon=True)
                        t.start()
                        alarm_playing = False
            else:
                counter   = 0
                is_drowsy = False

                # Simpan log setiap 15 frame
            if total_frame % 15 == 0 and ear_value > 0:
                confidence = get_confidence(ear_value)
                with open(LOG_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%H:%M:%S"),
                        status,
                        ear_value,
                        confidence
                    ])
    else:
        ear_value = 0.0
        status    = "Wajah tidak terdeteksi"
        color     = (150, 150, 150)

    # ── UI Overlay ──
    if last_result and last_result.face_landmarks:
        if is_drowsy:
            status = "MENGANTUK"
            color  = (0, 0, 255)

            winsound.Beep(1000, 300)

            # Overlay merah
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
            frame   = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)

            # Alert bar
            cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 180), -1)
            cv2.putText(frame, "DROWSINESS ALERT !",
                        (12, 48), cv2.FONT_HERSHEY_DUPLEX,
                        1.1, (255, 255, 255), 2)
        else:
            status = "NORMAL"
            color  = (0, 220, 100)

            cv2.rectangle(frame, (0, 0), (w, 50), (0, 80, 40), -1)
            cv2.putText(frame, "STATUS : NORMAL",
                        (12, 34), cv2.FONT_HERSHEY_DUPLEX,
                        0.85, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (0, 0), (w, 50), (40, 40, 40), -1)
        cv2.putText(frame, "Wajah tidak terdeteksi",
                    (12, 34), cv2.FONT_HERSHEY_DUPLEX,
                    0.75, (180, 180, 180), 1)

    # ── Info Panel Bawah ──
    inf_time  = round((time.time() - t_start) * 1000, 1)
    fps_list.append(1000 / inf_time if inf_time > 0 else 30)
    avg_fps   = round(sum(fps_list[-30:]) / len(fps_list[-30:]), 1)
    drowsy_pct = round(drowsy_frame / total_frame * 100, 1) if total_frame > 0 else 0

    cv2.rectangle(frame, (0, h - 60), (w, h), (15, 15, 15), -1)

    cv2.putText(frame, f"EAR: {ear_value}",
                (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 1)
    cv2.putText(frame, f"FPS: {avg_fps}",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(frame, f"Threshold: {EAR_THRESHOLD}",
                (175, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(frame, f"Drowsy: {drowsy_pct}%",
                (175, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(frame, f"Counter: {counter}/{FRAME_THRESHOLD}",
                (390, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(frame, "Q = Keluar",
                (390, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    # ── EAR Bar ──
    bx, by, bw, bh = w - 28, 55, 18, 280
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)
    fill   = int(bh * min(ear_value / 0.4, 1.0))
    bcolor = (0, 220, 100) if ear_value >= EAR_THRESHOLD else (0, 0, 255)
    cv2.rectangle(frame, (bx, by + bh - fill), (bx + bw, by + bh), bcolor, -1)
    ty = by + bh - int(bh * (EAR_THRESHOLD / 0.4))
    cv2.line(frame, (bx - 4, ty), (bx + bw + 4, ty), (255, 220, 0), 1)
    cv2.putText(frame, "EAR", (bx - 2, by - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    cv2.imshow("StayAwake AI — Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    

# ─────────────────────────────────────────
# RINGKASAN
# ─────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()

drowsy_pct = round(drowsy_frame / total_frame * 100, 1) if total_frame > 0 else 0
avg_fps    = round(sum(fps_list) / len(fps_list), 1) if fps_list else 0

print("\n" + "="*45)
print("📊 RINGKASAN SESI")
print("="*45)
print(f"Total Frame   : {total_frame}")
print(f"Frame Kantuk  : {drowsy_frame}")
print(f"% Kantuk      : {drowsy_pct}%")
print(f"Rata-rata FPS : {avg_fps}")
print("="*45)
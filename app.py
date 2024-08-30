import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import mediapipe as mp
from scipy.ndimage import label as ndi_label
import face_recognition
import psycopg2
from urllib.parse import urlparse
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time
from flask import Flask, request, jsonify

load_dotenv()

app = Flask(__name__)

# Load the TensorFlow Lite model for pill detection
interpreter = Interpreter(model_path='mobilenet_pill_detection.tflite')
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8)

# Initialize MediaPipe for person segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

img_size = (224, 224)

# Database connection setup
DATABASE_URL = "postgresql://cms_data_user:zD3zjXh6FRSd4GbInv0gALpHoCejfdCG@dpg-cr0aic3v2p9s73a4gpc0-a.singapore-postgres.render.com/cms_data"
result = urlparse(DATABASE_URL)
username = result.username
password = result.password
database = result.path[1:]
hostname = result.hostname
port = result.port

def get_db_connection():
    return psycopg2.connect(
        dbname=database,
        user=username,
        password=password,
        host=hostname,
        port=port
    )

def preprocess_for_pill_detection(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    lower_light = np.array([0, 0, 150])
    upper_light = np.array([180, 60, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_light = cv2.inRange(hsv, lower_light, upper_light)
    combined_mask = cv2.bitwise_or(mask_white, mask_light)
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    return combined_mask

def detect_pill(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    seg_results = selfie_segmentation.process(rgb_frame)
    person_mask = seg_results.segmentation_mask
    hand_results = hands.process(rgb_frame)
    hand_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(hand_mask, (x, y), 20, 255, -1)
    combined_mask = cv2.bitwise_and(person_mask, person_mask, mask=hand_mask)
    combined_mask = (combined_mask * 255).astype(np.uint8)
    roi = cv2.bitwise_and(frame, frame, mask=combined_mask)
    pill_mask = preprocess_for_pill_detection(roi)
    labeled_mask, num_labels = ndi_label(pill_mask)
    pill_detected = False
    for label_id in range(1, num_labels + 1):
        label_mask = (labeled_mask == label_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if 100 < area < 10000:
                x, y, w, h = cv2.boundingRect(largest_contour)
                roi = frame[y:y+h, x:x+w]
                img = cv2.resize(roi, img_size)
                img = img / 255.0
                img = np.expand_dims(img, axis=0).astype(np.float32)
                
                # Perform detection using TensorFlow Lite model
                interpreter.set_tensor(input_details[0]['index'], img)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])
                
                if prediction[0] > 0.9:
                    pill_detected = True
                    break
    return pill_detected

def update_medicine_intake(patient_id, medicine_time):
    conn = get_db_connection()
    cur = conn.cursor()
    current_date = datetime.now().date()
    cur.execute("""
        UPDATE medicine_intakes
        SET taken = TRUE
        WHERE patient_id = %s AND date = %s AND time = %s
    """, (patient_id, current_date, medicine_time))
    conn.commit()
    cur.close()
    conn.close()

def get_pending_medicine_times(patient_id):
    conn = get_db_connection()
    cur = conn.cursor()
    current_date = datetime.now().date()
    cur.execute("""
        SELECT time
        FROM medicine_intakes
        WHERE patient_id = %s AND date = %s AND taken = FALSE
        ORDER BY time
    """, (patient_id, current_date))
    pending_times = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return pending_times

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    patient_name = data.get('name')
    patient_ic = data.get('ic')
    if patient_name and patient_ic:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, image, medicine_times FROM patients WHERE name = %s AND ic = %s", (patient_name, patient_ic))
        result = cur.fetchone()
        cur.close()
        conn.close()
        if result:
            patient_id, patient_image, medicine_times = result
            if patient_image:
                nparr = np.frombuffer(patient_image, np.uint8)
                known_face_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                known_face_image = cv2.cvtColor(known_face_image, cv2.COLOR_BGR2RGB)
                known_face_encoding = face_recognition.face_encodings(known_face_image)[0]
                return jsonify({"success": True, "patient_id": patient_id, "known_face_encoding": known_face_encoding.tolist(), "medicine_times": medicine_times})
            else:
                return jsonify({"success": False, "error": "No image found for this patient."})
        else:
            return jsonify({"success": False, "error": "Invalid patient name or IC."})
    else:
        return jsonify({"success": False, "error": "Please provide both name and IC."})

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    frame = np.array(data['frame'], dtype=np.uint8)
    known_face_encoding = np.array(data['known_face_encoding'])
    patient_id = data['patient_id']

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    same_person_detected = False
    for face_encoding in face_encodings:
        results = face_recognition.compare_faces([known_face_encoding], face_encoding, tolerance=0.5)
        if results[0]:
            same_person_detected = True
            break

    pill_detected = False
    if same_person_detected:
        pill_detected = detect_pill(frame)

    pending_times = get_pending_medicine_times(patient_id)
    
    return jsonify({
        "same_person_detected": same_person_detected,
        "pill_detected": pill_detected,
        "pending_times": [time.strftime('%H:%M') for time in pending_times]
    })

@app.route('/update_intake', methods=['POST'])
def update_intake():
    data = request.json
    patient_id = data['patient_id']
    medicine_time = data['medicine_time']
    update_medicine_intake(patient_id, medicine_time)
    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

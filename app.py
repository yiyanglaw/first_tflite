import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import psycopg2
from urllib.parse import urlparse
from datetime import datetime, timedelta
import logging
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the inference client for pill detection
PILL_CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="lJ0tgYHt58Kl2kCeMlQb"
)

# Initialize mediapipe pose and hand detectors
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define the threshold for hand near mouth detection
HAND_NEAR_MOUTH_THRESHOLD = 1.8

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

def detect_pills(frame):
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_filename = temp_file.name
        cv2.imwrite(temp_filename, frame)
        result = PILL_CLIENT.infer(temp_filename, model_id="pill-detection-llp4r/3")
    os.remove(temp_filename)

    boxes = []
    if 'predictions' in result:
        predictions = result['predictions']
        for pred in predictions:
            if pred['confidence'] > 0.62:
                x = int(pred['x'] - pred['width'] / 2)
                y = int(pred['y'] - pred['height'] / 2)
                w = int(pred['width'])
                h = int(pred['height'])
                confidence = pred['confidence']
                class_name = pred['class']
                boxes.append({'x': x, 'y': y, 'width': w, 'height': h, 'confidence': confidence, 'class': class_name})
    return boxes

def detect_bottles(frame):
    # Load YOLO
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4 and classes[class_id] == 'bottle':
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_bottles = []
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            detected_bottles.append({
                'box': boxes[i],
                'confidence': confidences[i],
                'class': classes[class_ids[i]]
            })
    return detected_bottles

def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

def check_hand_near_mouth(pose_landmarks, hand_landmarks):
    if pose_landmarks and hand_landmarks:
        mouth_left = pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]
        mouth_right = pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT]

        for hand_landmark in hand_landmarks:
            index_tip = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmark.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_distance_left = calculate_distance(index_tip, mouth_left)
            index_distance_right = calculate_distance(index_tip, mouth_right)
            thumb_distance_left = calculate_distance(thumb_tip, mouth_left)
            thumb_distance_right = calculate_distance(thumb_tip, mouth_right)

            if (0.65 < index_distance_left < HAND_NEAR_MOUTH_THRESHOLD or
                0.65 < index_distance_right < HAND_NEAR_MOUTH_THRESHOLD or
                0.65 < thumb_distance_left < HAND_NEAR_MOUTH_THRESHOLD or
                0.65 < thumb_distance_right < HAND_NEAR_MOUTH_THRESHOLD):
                return True
    return False

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    patient_name = data.get('name')
    patient_ic = data.get('ic')
    if patient_name and patient_ic:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, image FROM patients WHERE name = %s AND ic = %s", (patient_name, patient_ic))
        result = cur.fetchone()
        cur.close()
        conn.close()
        if result:
            patient_id, patient_image = result
            if patient_image:
                nparr = np.frombuffer(patient_image, np.uint8)
                known_face_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                known_face_features = cv2.resize(known_face_image, (100, 100)).flatten().tolist()
                return jsonify({"success": True, "patient_id": patient_id, "known_face_features": known_face_features})
            else:
                return jsonify({"success": False, "message": "No image found for this patient."})
        else:
            return jsonify({"success": False, "message": "Invalid patient name or IC."})
    else:
        return jsonify({"success": False, "message": "Please provide both name and IC."})

@app.route('/detect_pill', methods=['POST'])
def process_image():
    file = request.files['image']
    npimg = np.fromfile(file, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # Detect pills
    pill_boxes = detect_pills(frame)
    pill_detected = len(pill_boxes) > 0

    # Detect pose and hands
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(image)
        hand_results = hands.process(image)

    # Check for hand near mouth
    hand_near_mouth = check_hand_near_mouth(pose_results.pose_landmarks, hand_results.multi_hand_landmarks)

    # Detect bottles
    bottle_boxes = detect_bottles(frame)
    bottle_detected = len(bottle_boxes) > 0

    return jsonify({
        "pill_detected": pill_detected,
        "hand_near_mouth": hand_near_mouth,
        "bottle_detected": bottle_detected
    })

@app.route('/update_intake', methods=['POST'])
def update_intake():
    data = request.json
    patient_id = data.get('patient_id')
    medicine_time = data.get('medicine_time')
    
    if not patient_id or not medicine_time:
        return jsonify({"success": False, "message": "Missing patient_id or medicine_time"}), 400
    
    conn = get_db_connection()
    cur = conn.cursor()
    current_date = datetime.now().date()
    try:
        medicine_time_obj = datetime.strptime(medicine_time, '%I:%M %p').time()
        
        cur.execute("""
            UPDATE medicine_intakes
            SET taken = TRUE
            WHERE patient_id = %s AND date = %s AND time = %s AND taken = FALSE
            RETURNING id
        """, (patient_id, current_date, medicine_time_obj))
        updated_row = cur.fetchone()
        conn.commit()
        
        if updated_row:
            logging.info(f"Updated intake for patient {patient_id} at {medicine_time}")
            return jsonify({"success": True, "message": "Medicine intake recorded."})
        else:
            logging.warning(f"No matching untaken intake found for patient {patient_id} at {medicine_time}")
            return jsonify({"success": False, "message": "No matching untaken intake found."}), 404
    except Exception as e:
        conn.rollback()
        logging.error(f"Error updating intake: {str(e)}")
        return jsonify({"success": False, "message": f"Database error occurred: {str(e)}"}), 500
    finally:
        cur.close()
        conn.close()

@app.route('/get_pending_times/<int:patient_id>', methods=['GET'])
def get_pending_times(patient_id):
    conn = get_db_connection()
    cur = conn.cursor()
    current_date = datetime.now().date()
    try:
        cur.execute("""
            SELECT time
            FROM medicine_intakes
            WHERE patient_id = %s AND date = %s AND taken = FALSE
            ORDER BY time
        """, (patient_id, current_date))
        pending_times = [row[0].strftime('%I:%M %p') for row in cur.fetchall()]
        logging.info(f"Retrieved pending times for patient {patient_id}: {pending_times}")
        return jsonify({"pending_times": pending_times})
    except Exception as e:
        logging.error(f"Error retrieving pending times: {str(e)}")
        return jsonify({"error": f"Database error occurred: {str(e)}"}), 500
    finally:
        cur.close()
        conn.close()

@app.route('/check_new_day/<int:patient_id>', methods=['GET'])
def check_new_day(patient_id):
    conn = get_db_connection()
    cur = conn.cursor()
    current_date = datetime.now().date()
    try:
        cur.execute("""
            SELECT COUNT(*) FROM medicine_intakes
            WHERE patient_id = %s AND date = %s
        """, (patient_id, current_date))
        count = cur.fetchone()[0]
        
        if count == 0:
            cur.execute("""
                INSERT INTO medicine_intakes (patient_id, date, time, taken)
                SELECT %s, %s, unnest(medicine_times), FALSE
                FROM patients
                WHERE id = %s
            """, (patient_id, current_date, patient_id))
            conn.commit()
            
            cur.execute("""
                SELECT time
                FROM medicine_intakes
                WHERE patient_id = %s AND date = %s AND taken = FALSE
                ORDER BY time
            """, (patient_id, current_date))
            pending_times = [row[0].strftime('%I:%M %p') for row in cur.fetchall()]
            
            return jsonify({"new_day": True, "pending_times": pending_times})
        else:
            return jsonify({"new_day": False})
    except Exception as e:
        conn.rollback()
        logging.error(f"Error checking new day: {str(e)}")
        return jsonify({"error": f"Database error occurred: {str(e)}"}), 500
    finally:
        cur.close()
        conn.close()

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7777)

import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from tflite_runtime.interpreter import Interpreter
import psycopg2
from urllib.parse import urlparse
from datetime import datetime

app = Flask(__name__)

# Load the TensorFlow Lite model for pill detection
interpreter = Interpreter(model_path='mobilenet_pill_detection.tflite')
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data or 'patient_id' not in data:
        return jsonify({"error": "Invalid request data"}), 400

    image_data = np.frombuffer(data['image'].encode('latin1'), dtype=np.uint8)
    frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    img_size = (224, 224)

    pill_mask = preprocess_for_pill_detection(frame)
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

                label_text = 'Pill' if prediction[0] > 0.9 else 'No Pill'
                pill_detected = label_text == 'Pill'
    
    if pill_detected:
        conn = get_db_connection()
        cur = conn.cursor()
        current_date = datetime.now().date()
        cur.execute("""
            UPDATE medicine_intakes
            SET taken = TRUE
            WHERE patient_id = %s AND date = %s AND time = %s
        """, (data['patient_id'], current_date, data['medicine_time']))
        conn.commit()
        cur.close()
        conn.close()

    return jsonify({"pill_detected": pill_detected})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

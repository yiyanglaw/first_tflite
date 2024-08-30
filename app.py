import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import mediapipe as mp
from scipy.ndimage import label as ndi_label
import face_recognition
from flask import Flask, request, jsonify

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
                
                pill_detected = prediction[0] > 0.9
    return pill_detected

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    frame = np.array(data['frame'], dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    
    known_face_encoding = np.array(data['known_face_encoding'])
    face_encodings = face_recognition.face_encodings(frame)
    
    same_person_detected = any(face_recognition.compare_faces([known_face_encoding], encoding, tolerance=0.5) for encoding in face_encodings)
    
    if same_person_detected:
        pill_detected = detect_pill(frame)
        if pill_detected:
            return jsonify({"result": "Pill detected"})
        return jsonify({"result": "Pill not detected"})
    return jsonify({"result": "Face not recognized"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

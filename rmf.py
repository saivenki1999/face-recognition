import face_recognition as fr
import cv2
import numpy as np
import os
import threading
import json
import requests
from flask import Flask, render_template
import datetime
from PIL import Image

# Telegram configuration
TELEGRAM_BOT_TOKEN = "8489823981:AAHO9EfkPgyLYaIlJKjxEqKyiFHqBwFc3Ag"
TELEGRAM_CHAT_ID = "2032431113" 

# Initialize Flask app
app = Flask(__name__)
detected_faces = []
unknown_alert_sent = False

# Function to send Telegram alert
def send_telegram_alert():
    global unknown_alert_sent
    message = "🚨 ALERT: Unknown Person Detected!\nCheck the Police Surveillance Server for details."
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        params = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message
        }
        response = requests.post(url, params=params)
        if response.status_code == 200:
            print("Telegram alert sent successfully!")
            unknown_alert_sent = True
        else:
            print(f"Failed to send Telegram alert. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error sending Telegram alert: {e}")

# Load known faces and their details
path = "./train/"
known_names = []
known_name_encodings = []
known_details = {}
images = os.listdir(path)

for img_name in images:

    image = Image.open(os.path.join(path, img_name)).convert("RGB")
    image = np.array(image)

    encodings = fr.face_encodings(image)

    if len(encodings) == 0:
        print("No face found:", img_name)
        continue

    known_name_encodings.append(encodings[0])
    name = os.path.splitext(img_name)[0].capitalize()
    known_names.append(name)
        
        # Load person details
json_path = os.path.join("known_persons", f"{name.lower()}.json")
if os.path.exists(json_path):
            with open(json_path) as f:
                known_details[name] = json.load(f)
else:
            known_details[name] = {
                "parents": "Not Available",
                "address": "Not Available"
            }

print("Loaded known faces:", known_names)

def face_recognition_loop():
    global unknown_alert_sent
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = fr.face_locations(rgb_frame)
        face_encodings = fr.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = fr.compare_faces(known_name_encodings, face_encoding)
            name = "Unknown"
            parents = "N/A"
            address = "N/A"

            face_distances = fr.face_distance(known_name_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_names[best_match_index]
                unknown_alert_sent = False
                # Get additional details
                details = known_details.get(name, {})
                parents = details.get("parents", "N/A")
                address = details.get("address", "N/A")

            # Capture & save image
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            image_path = f'static/faces/{name}_{timestamp}.jpg'
            os.makedirs("static/faces", exist_ok=True)
            cv2.imwrite(image_path, frame)

            # Save detected person data
            detected_faces.append({
                "name": name,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_path": image_path,
                "parents": parents,
                "address": address

            })

            # Send Telegram alert for unknown faces
            if name == "Unknown" and not unknown_alert_sent:
                send_telegram_alert()

            # Draw rectangle and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow("Real-Time Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

@app.route('/')
def home():
    unknown_detected = any(face['name'] == 'Unknown' for face in detected_faces)
    return render_template(
        'index.html',
        detected_faces=detected_faces,
        unknown_detected=unknown_detected
    )

if __name__ == "__main__":
    face_thread = threading.Thread(target=face_recognition_loop, daemon=True)
    face_thread.start()
    app.run(debug=False, host="0.0.0.0", port=5000)
import face_recognition
import cv2
import numpy as np
import os

train_path = "train"

known_encodings = []
known_names = []

# LOAD TRAINING IMAGES
for file in os.listdir(train_path):

    img_path = os.path.join(train_path, file)

    img = cv2.imread(img_path)

    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_encodings(img)

    if len(faces) == 0:
        print("No face in:", file)
        continue

    known_encodings.append(faces[0])
    known_names.append(os.path.splitext(file)[0])

print("Loaded faces:", known_names)

# LOAD TEST IMAGE
test_img = cv2.imread("test/test.jpg")

test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

locations = face_recognition.face_locations(test_img)
encodings = face_recognition.face_encodings(test_img, locations)

for (top, right, bottom, left), face_encoding in zip(locations, encodings):

    matches = face_recognition.compare_faces(known_encodings, face_encoding)

    name = "Unknown"

    distances = face_recognition.face_distance(known_encodings, face_encoding)

    best = np.argmin(distances)

    if matches[best]:
        name = known_names[best]

    cv2.rectangle(test_img,(left,top),(right,bottom),(0,255,0),2)
    cv2.putText(test_img,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)

cv2.imshow("Result", test_img)

cv2.imwrite("output.jpg", test_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
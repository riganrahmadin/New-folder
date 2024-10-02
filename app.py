from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the Haar cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load pre-trained models
age_model = load_model('age_model.h5')
emotion_model = load_model('emotion_model.h5')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the webcam
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_color = frame[y:y + h, x:x + w]
            roi_gray = gray[y:y + h, x:x + w]
            roi_color_resized = cv2.resize(roi_color, (48, 48))

            # Age Prediction
            age_input = cv2.resize(roi_color, (200, 200)) / 255.0
            age_input = np.expand_dims(age_input, axis=0)
            predicted_age = age_model.predict(age_input)
            age = int(predicted_age[0][0])  # Assuming model returns a single value

            # Emotion Prediction
            emotion_input = roi_gray.astype('float32') / 255.0
            emotion_input = cv2.resize(emotion_input, (48, 48))
            emotion_input = np.expand_dims(emotion_input, axis=0)
            emotion_input = np.expand_dims(emotion_input, axis=-1)
            predicted_emotion = emotion_model.predict(emotion_input)
            emotion = emotion_labels[np.argmax(predicted_emotion)]

            # Draw rectangles and labels
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f'Age: {age}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f'Emotion: {emotion}', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

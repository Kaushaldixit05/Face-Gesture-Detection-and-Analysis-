import cv2
import mediapipe as mp
from deepface import DeepFace
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# Video Capture
cap = cv2.VideoCapture(0)

# Variables for tracking
total_looks_away = 0
look_away_frames = 0
total_happy = 0
total_neutral = 0
total_sad = 0
total_angry = 0

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the BGR image to RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Face Mesh processing
    results = face_mesh.process(rgb_frame)
    
    is_looking_away = False
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Manually draw landmarks
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            # Get eye landmarks
            left_eye_landmarks = face_landmarks.landmark[133:144]
            right_eye_landmarks = face_landmarks.landmark[362:373]
            
            # Calculate the average x-coordinate for the eyes
            left_eye_x = sum([landmark.x for landmark in left_eye_landmarks]) / len(left_eye_landmarks)
            right_eye_x = sum([landmark.x for landmark in right_eye_landmarks]) / len(right_eye_landmarks)
            
            # If eyes are looking away (based on x-coordinate threshold)
            if left_eye_x > 0.6 or right_eye_x > 0.6:
                look_away_frames += 1
                is_looking_away = True
            else:
                if look_away_frames > 5:
                    total_looks_away += 1
                look_away_frames = 0
    
    # Analyze facial expression
    try:
        emotions = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = emotions['dominant_emotion']
        
        if dominant_emotion == 'happy':
            total_happy += 1
        elif dominant_emotion == 'neutral':
            total_neutral += 1
        elif dominant_emotion == 'sad':
            total_sad += 1
        elif dominant_emotion == 'angry':
            total_angry += 1
    except:
        pass
    
    # If looking away, display a red warning on the screen
    if is_looking_away:
        cv2.putText(frame, 'Warning: Looking Away!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Confidence Tracker', frame)
    
    # Exit on pressing 'Escape'
    if cv2.waitKey(5) & 0xFF == 27:  # Escape key
        break

# Calculate summary
duration = time.time() - start_time
total_frames = duration * cap.get(cv2.CAP_PROP_FPS)

confidence_summary = f"""
Summary of the Interview:
- Total duration: {int(duration)} seconds
- Times looked away: {total_looks_away}
- Happy expressions: {total_happy} times
- Neutral expressions: {total_neutral} times
- Sad expressions: {total_sad} times
- Angry expressions: {total_angry} times
- Confidence level: {'High' if total_looks_away < 5 and total_happy > total_sad + total_angry else 'Low'}
"""

# Release resources
cap.release()
cv2.destroyAllWindows()

# Print the summary
print(confidence_summary)

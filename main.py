import cv2
from deepface import DeepFace

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    try:
        # Analyze frame with DeepFace
        results = DeepFace.analyze(
            frame,
            actions=['emotion', 'gender', 'race', 'age'],
            detector_backend='retinaface',
            enforce_detection=True
        )

        # Extract results
        result = results[0]
        emotion = result['dominant_emotion']
        gender = result['dominant_gender']
        race = result['dominant_race']
        age = result['age']

        # Overlay info on frame
        text = f"{emotion}, {gender}, {race}, Age: {age}"
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    except Exception as e:
        cv2.putText(frame, "No face detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # Show the frame
    cv2.imshow('Real-Time Face Analysis', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
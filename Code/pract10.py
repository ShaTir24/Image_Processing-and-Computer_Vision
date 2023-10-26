import cv2

# Load the pre-trained face detection model
faceCascade = cv2.CascadeClassifier('./Face_detection/haarcascade_frontalface_default.xml')

# Open the video file for reading
cap = cv2.VideoCapture('./Media/input_face.mp4')

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Performing face detection
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Drawing rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Displaying the frame with rectangles drawn around faces
    cv2.imshow('Video', frame)
    # Exit when the 'q' key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break        

cap.release()
cv2.destroyAllWindows()
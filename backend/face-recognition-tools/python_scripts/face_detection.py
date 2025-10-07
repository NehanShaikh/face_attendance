from mtcnn import MTCNN
import cv2

detector = MTCNN()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    result = detector.detect_faces(frame)
    for face in result:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

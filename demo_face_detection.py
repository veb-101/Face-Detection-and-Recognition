import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 640)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,  # frame
        scaleFactor=1.12,  # img size reduce at each scale
        minNeighbors=5,  # neighbors each candidate rectangles should retain
        minSize=(20, 20)  # min threshold size, less than this -> ignore
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+w),
                      (255, 0, 0), 2)

        # draw on both frames
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    cv2.imshow('video', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

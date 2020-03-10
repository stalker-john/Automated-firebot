import numpy as np
import cv2

fire_cascade = cv2.CascadeClassifier('fire_detection.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fire = fire_cascade.detectMultiScale(gray, 1.2, 5)
    # roi_gray = gray[y:y+h, x:x+w]
    # roi_color = img[y:y+h, x:x+w]

    for (x,y,w,h) in fire:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        

    cv2.imshow("img",img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
    

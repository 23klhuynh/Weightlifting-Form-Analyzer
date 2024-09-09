import cv2
import numpy as np
import poseModule as pm

cap = cv2.VideoCapture(0)
detector = pm.poseDetector()
count = 0
dir = 0

while True:
    success, img = cap.read()

    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)

    if len(lmList) != 0:
        angle, elbow_angle = detector.findAngle(img)
        per = np.interp(angle, (30, 170), (0,100))

        #check for the dumbell curls
        if per == 100:
            if dir == 0:
                count += 0.5
                dir = 1
                
        if per == 0:
            if dir == 1:
                count += 0.5
                dir = 0

        #feedback
        if per < 100 and int(count) > 0:
            if dir == 0:
                cv2.putText(img, "All the way down",(500, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 3)
            else:
                cv2.putText(img, "All the way up",(500, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 3)

        if elbow_angle > 20:
            cv2.putText(img, "Warning:", (0, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 3)
            cv2.putText(img, "Elbow Out of Position", (0, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 3)
        else:
            if int(count) > 0:
                cv2.putText(img, "Good Form!", (500, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        #end

        cv2.rectangle(img, (0, 0), (100, 100), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (25, 75), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 5)

    cv2.imshow("image", img) 
    if (cv2.waitKey(10) == ord("q")):
        break

cap.release()
cv2.destroyAllWindows()

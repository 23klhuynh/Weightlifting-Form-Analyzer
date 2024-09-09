import cv2 
import mediapipe as mp
import math
import numpy as np

#cap = cv2.VideoCapture("/Users/khanghuynh/Desktop/py_script/AI_trainer/assets/6.MOV")#3.MOV
cap = cv2.VideoCapture(0)
mpPose = mp.solutions.pose
pose = mpPose.Pose()
draw = mp.solutions.drawing_utils

dir = 0
count = 0

while cap.isOpened():
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        lmList = []
        landmark = results.pose_landmarks.landmark
        
        for id, lm in enumerate(landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])
        #///////////////////////////////

        left_side_landmarks = [
            landmark[mpPose.PoseLandmark.LEFT_SHOULDER.value],
            landmark[mpPose.PoseLandmark.LEFT_ELBOW.value],
            landmark[mpPose.PoseLandmark.LEFT_WRIST.value],
            landmark[mpPose.PoseLandmark.LEFT_HIP.value],
            landmark[mpPose.PoseLandmark.LEFT_KNEE.value],
            landmark[mpPose.PoseLandmark.LEFT_ANKLE.value]
        ]

        right_side_landmarks = [
            landmark[mpPose.PoseLandmark.RIGHT_SHOULDER.value],
            landmark[mpPose.PoseLandmark.RIGHT_ELBOW.value],
            landmark[mpPose.PoseLandmark.RIGHT_WRIST.value],
            landmark[mpPose.PoseLandmark.RIGHT_HIP.value],
            landmark[mpPose.PoseLandmark.RIGHT_KNEE.value],
            landmark[mpPose.PoseLandmark.RIGHT_ANKLE.value]
        ]

        left_side_confidence = sum([lm.visibility for lm in left_side_landmarks]) / len(left_side_landmarks)

        right_side_confidence = sum([lm.visibility for lm in right_side_landmarks]) / len(right_side_landmarks)

        if left_side_confidence > right_side_confidence:
            #left side
            elbow_x, elbow_y = lmList[13][1:]
            hand_x, hand_y = lmList[15][1:]
            shoulder_x, shoulder_y = lmList[11][1:]
            hip_x, hip_y = lmList[23][1:]
            knee_x, knee_y = lmList[25][1:]
            foot_x, foot_y = lmList[27][1:]
        else:
            #right side
            elbow_x, elbow_y = lmList[14][1:]
            hand_x, hand_y = lmList[16][1:]
            shoulder_x, shoulder_y = lmList[12][1:]
            hip_x, hip_y = lmList[24][1:]
            knee_x, knee_y = lmList[26][1:]
            foot_x, foot_y = lmList[28][1:]
            

        #///////////////////////////////////////
        #point1 knee
        hip_knee_vec = (knee_x-hip_x, knee_y-hip_y)
        knee_foot_vec = (knee_x-foot_x, knee_y-foot_y)
        knee_prd = (hip_knee_vec[0]*knee_foot_vec[0]) + (hip_knee_vec[1]*knee_foot_vec[1])
        knee_hip_mag = math.sqrt(hip_knee_vec[0]**2 + hip_knee_vec[1]**2)
        knee_foot_mag = math.sqrt(knee_foot_vec[0]**2 + knee_foot_vec[1]**2)
        division1 = knee_prd / (knee_hip_mag * knee_foot_mag)
        knee_angle = math.degrees(math.acos(division1))

        #point2 hip
        hip_shoulder_vec = (shoulder_x-hip_x, shoulder_y-hip_y)
        hip_knee_vec = (knee_x-hip_x, knee_y-hip_y)
        hip_prd = (hip_shoulder_vec[0]*hip_knee_vec[0]) + (hip_shoulder_vec[1]*hip_knee_vec[1])
        hip_shoulder_mag = math.sqrt(hip_shoulder_vec[0]**2 + hip_shoulder_vec[1]**2)
        hip_knee_mag = math.sqrt(hip_knee_vec[0]**2 + hip_knee_vec[1]**2)
        division2 = hip_prd / (hip_shoulder_mag * hip_knee_mag)
        hip_angle = math.degrees(math.acos(division2))

        shoulder_elbow_vec = (elbow_x-shoulder_x, elbow_y-shoulder_y)
        shoulder_angle = abs(90 - math.degrees(math.atan2(shoulder_elbow_vec[1], shoulder_elbow_vec[0])))

        cv2.line(img, (shoulder_x, shoulder_y), (elbow_x, elbow_y), (255, 0, 255), 3)
        cv2.line(img, (elbow_x, elbow_y), (hand_x, hand_y), (255, 0, 255), 3)

        cv2.line(img, (shoulder_x,shoulder_y), (hip_x, hip_y), (255, 0, 255), 3)
        cv2.line(img, (hip_x,hip_y), (knee_x, knee_y), (255, 0, 255), 3)
        cv2.line(img, (foot_x,foot_y), (knee_x, knee_y), (255, 0, 255), 3)
        cv2.circle(img, (shoulder_x, shoulder_y), 5, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (hip_x, hip_y), 5, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (knee_x, knee_y), 5, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (foot_x, foot_y), 5, (255, 0, 0), cv2.FILLED)

        #90 degree
        cv2.line(img, (shoulder_x, shoulder_y), (shoulder_x, shoulder_y+200), (0, 0, 255), 3)
        cv2.putText(img, str(int(knee_angle)), (knee_x-25, knee_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, str(int(hip_angle)), (hip_x+50, hip_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, str(int(shoulder_angle)), (shoulder_x, shoulder_y-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        per = np.interp(hip_angle, (45, 172), (0, 100))

        if per == 100:
            if dir == 0:
                count += 0.5
                dir = 1
            
        if per == 0:
            if dir == 1:
                count += 0.5
                dir = 0

        cv2.putText(img, str(int(count)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
        cv2.putText(img, f'{int(per)}%', (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)

        # feedback
        # arm

        if per < 100:
            if shoulder_angle > 18:
                cv2.putText(img, "Warning: Arm Position Incorrect", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # prevent backbending
            if per < 35 and knee_angle > 125:
                cv2.putText(img, "warning: Knee position Incorrect", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if (per > 30 and per < 60 and (knee_angle < 90 or knee_angle > 150)) or (per > 60 and knee_angle < 150):
                cv2.putText(img, "warning: Back bending", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("deadlift", img)

    if cv2.waitKey(10) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

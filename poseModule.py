import cv2
import mediapipe as mp
import math

class poseDetector():

    def __init__(self, mode=False, model_complexity=1, smooth_landmarks=True, detectionCon=0.5, trackCon=0.5):
        
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode, model_complexity=self.model_complexity, smooth_landmarks=self.smooth_landmarks, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        self.landmarks = self.results.pose_landmarks

        if self.landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img
        
    def findPosition(self, img, draw=True):
        self.lmList = []

        if self.landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id,cx,cy])
                if draw: 
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        
        return self.lmList
    
    def findAngle(self, img, draw=True):
        left_arm_landmarks = [
            self.landmarks.landmark[self.mpPose.PoseLandmark.LEFT_ELBOW.value],
            self.landmarks.landmark[self.mpPose.PoseLandmark.LEFT_WRIST.value],
            self.landmarks.landmark[self.mpPose.PoseLandmark.LEFT_SHOULDER.value],
        ]

        right_arm_landmarks = [
            self.landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_ELBOW.value],
            self.landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_WRIST.value],
            self.landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_SHOULDER.value],
        ]

        left_arm_confidence = sum([lm.visibility for lm in left_arm_landmarks]) / len(left_arm_landmarks)

        right_arm_confidence = sum([lm.visibility for lm in right_arm_landmarks]) / len(right_arm_landmarks)

        if left_arm_confidence > right_arm_confidence:
            x1, y1 = self.lmList[11][1:]
            x2, y2 = self.lmList[13][1:]
            x3, y3 = self.lmList[15][1:]
        else:
            x1, y1 = self.lmList[12][1:]
            x2, y2 = self.lmList[14][1:]
            x3, y3 = self.lmList[16][1:]
        # get the landmarks
        """ x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:] """
        # 180 degree ///////////////////////////////
        vec1 = (x1 - x2, y1 - y2)  
        vec2 = (x3 - x2, y3 - y2)  
        # calculate dot product and magnitudes
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        mag_vec1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
        mag_vec2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)
        cos_angle = dot_product / (mag_vec1 * mag_vec2)
        angle = math.degrees(math.acos(cos_angle))
        
        #///////////////////////
        #elbow degree
        del_x = x2-x1
        del_y = y2-y1
        elbow_angle = abs(math.degrees(math.atan2(del_x, del_y)))
        
        # draw
        if draw:
            cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 3)
            cv2.line(img, (x3,y3), (x2,y2), (0,255,0), 3)
            cv2.circle(img, (x1,y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1,y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2,y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2,y2), 15, (255, 0, 255), 2)
            cv2.circle(img, (x3,y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3,y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2-20, y2+100), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 255), 2)

            #draw 90 degree
            cv2.line(img, (x1, y1), (x1, y1+100), (0,0,255), 3)
            cv2.putText(img, str(int(elbow_angle)),(x1, y1+50), cv2.FONT_HERSHEY_PLAIN, 4, (255,0,255), 2)

        return angle, elbow_angle

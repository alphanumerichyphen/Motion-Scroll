import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict


class HandDetector:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):
        """Initializes a MediaPipe Hand object."""
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findhands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findposition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                xList.append(cx)
                yList.append(cy)

                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 6, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)
        return self.lmList, bbox

    def fingersup(self):
        fingers = []
        tipIds = [4, 8, 12, 16, 20]

        # Thumb
        for idx, hand_handedness in enumerate(self.results.multi_handedness):
            handedness_dict = MessageToDict(hand_handedness)

        if handedness_dict['classification'][0]['index'] == 0:  # LEFT HAND
            if self.lmList[tipIds[0]][1] > self.lmList[tipIds[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:  # RIGHT HAND
            if self.lmList[tipIds[0]][1] < self.lmList[tipIds[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # 4 fingers
        for id in range(1, 5):
            if self.lmList[tipIds[id]][2] < self.lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 1280, 960
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "Numbers"
myList = os.listdir(folderPath)
print(myList)
overlaylist = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlaylist.append(image)

print(len(overlaylist))
pTime = 0

detector = htm.handDetector(detectionCon=0.8)


tipIDs = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[4][1] < lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if lmList[tipIDs[id]][2] < lmList[tipIDs[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = overlaylist[totalFingers -1].shape
        img[0:h, 0:w] = overlaylist[totalFingers -1]

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (800, 60), cv2.FONT_HERSHEY_SIMPLEX,
                2, (100, 150, 100), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
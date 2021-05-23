import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
from pynput.mouse import Button, Controller

####################################
wcam, hcam = 640, 480
frameReduction = 120
smoothening = 7
####################################

plocx, plocy = 0, 0
clocx, clocy = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)
pTime = 0

mouse = Controller()

detector = htm.HandDetector(maxHands=1, detectionCon=0.8)
wscreen, hscreen = autopy.screen.size()
print(wscreen, hscreen)  # screen width and height

while True:
    # 1 find handLandmarks
    success, img = cap.read()
    img = detector.findhands(img)
    lmList, bbox = detector.findposition(img)

    # 2 Tip of the fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # co-ordinates of first fingertip
        x2, y2 = lmList[12][1:]  # co-ordinates of second fingertip

        # 3 Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # show effective area of mouse movement
        cv2.rectangle(img, (frameReduction, frameReduction-50), (wcam - frameReduction, hcam - frameReduction - 50),
                      (255, 0, 255), 2)

        # 4 Only first finger -> Moving mode
        if fingers[1] == 1 and fingers[2] == 0:

            # 5 Convert coordinates 640x480 to 1536x864
            x3 = np.interp(x1, (frameReduction, wcam-frameReduction), (0, wscreen))
            y3 = np.interp(y1, (frameReduction-50, hcam-frameReduction-50), (0, hscreen))

            # 6 Smoothen values
            clocx = plocx + (x3 - plocx) / smoothening
            clocy = plocy + (y3 - plocy) / smoothening

            # 7 Move mouse
            autopy.mouse.move(wscreen - clocx, clocy)
            cv2.circle(img, (x1, y1), 15, (255, 255, 0), cv2.FILLED)  # circle on first fingertip
            plocx, plocy = clocx, clocy

            # 8 Thumb down ----> Clicking mode
            if fingers[0] == 1:
                # Click when thumb down
                # cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
                autopy.mouse.click()

        # First and second finger -----> Scrolling Mode
        elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
            if fingers[0] == 1:
                mouse.scroll(0, 0.05)

            if fingers[0] == 0:
                mouse.scroll(0, -0.05)

    # 10 frame rate
    cTime = time.time()
    if cTime-pTime > 0:
        fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
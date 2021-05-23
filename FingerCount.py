import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# image output
folderPath = "Numbers"
myList = os.listdir(folderPath)
# print(myList) images in folder
overlaylist = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlaylist.append(image)
# print(len(overlaylist)) number of images

pTime = 0
detector = htm.HandDetector(detectionCon=0.7)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findhands(img)
    lmList, bbox = detector.findposition(img)

    if len(lmList) != 0:
        fingers = detector.fingersup()
        totalFingers = fingers.count(1)  # how many ones i.e. fingers up
        # print(totalFingers)

        h, w, c = overlaylist[totalFingers - 1].shape  # which image to choose from folder
        img[0:h, 0:w] = overlaylist[totalFingers - 1]  # where to put image

    cTime = time.time()
    if cTime - pTime > 0:
        fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (800, 60), cv2.FONT_HERSHEY_SIMPLEX,
                2, (100, 150, 100), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
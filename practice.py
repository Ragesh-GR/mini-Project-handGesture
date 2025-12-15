import cv2
import numpy as np
import HandTrackingModule as htm
import time
import pyautogui

##########################
wCam, hCam = 640, 480
frameR = 100        # Frame Reduction
smoothening = 7
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# FIX 1: Camera index changed from 1 → 0
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)

# FIX 2: autopy replaced with pyautogui
wScr, hScr = pyautogui.size()

while True:
    success, img = cap.read()
    if not success:
        continue

    # 1. Find hand landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # Draw movement frame
    cv2.rectangle(
        img, (frameR, frameR),
        (wCam - frameR, hCam - frameR),
        (255, 0, 255), 2
    )

    if lmList:
        # 2. Get index & middle finger tips
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # 4. Only index finger up → Move mouse
        if fingers[1] == 1 and fingers[2] == 0:
            # Convert coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # Smooth movement
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            pyautogui.moveTo(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

            plocX, plocY = clocX, clocY

        # 5. Index + Middle fingers → Click
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)

            if length < 40:
                cv2.circle(
                    img, (lineInfo[4], lineInfo[5]),
                    15, (0, 255, 0), cv2.FILLED
                )
                pyautogui.click()
                time.sleep(0.2)

    # 6. FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(
        img, f'FPS: {int(fps)}',
        (20, 50),
        cv2.FONT_HERSHEY_PLAIN, 3,
        (255, 0, 0), 3
    )

    cv2.imshow("Hand Gesture Mouse", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
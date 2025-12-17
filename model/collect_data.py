import cv2
import csv
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
detector = htm.handDetector(maxHands=1)

labels = ['S', 'R', 'M', 'X', 'V']
current_label = 'S'   # change manually

file = open('dataset/data.csv', 'a', newline='')
writer = csv.writer(file)

print("Press 's','r','m','x','v' to change label")
print("Press 'q' to quit")

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, _ = detector.findPosition(img)

    if lmList:
        row = []
        for lm in lmList:
            row.extend(lm[1:])  # x,y only
        row.append(current_label)
        writer.writerow(row)
        print("Saved:", current_label)

    cv2.imshow("Collect Data", img)

    key = cv2.waitKey(1)
    if key == ord('s'): current_label = 'S'
    if key == ord('r'): current_label = 'R'
    if key == ord('m'): current_label = 'M'
    if key == ord('x'): current_label = 'X'
    if key == ord('v'): current_label = 'V'
    if key == ord('q'): break

cap.release()
file.close()
cv2.destroyAllWindows()

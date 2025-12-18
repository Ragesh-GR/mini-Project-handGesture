import cv2
import time
import math
import numpy as np
import HandTrackingModule as htm
import pyautogui
import subprocess

# Configuration
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(maxHands=1, detectionCon=0.85, trackCon=0.8)

# Volume control for Linux
def get_volume():
    """Get current system volume (0-100)"""
    try:
        output = subprocess.check_output(['amixer', 'get', 'Master']).decode()
        import re
        match = re.search(r'\[(\d+)%\]', output)
        if match:
            return int(match.group(1))
    except:
        pass
    return 50

def set_volume(percent):
    """Set system volume smoothly (0-100)"""
    try:
        percent = max(0, min(100, int(percent)))
        subprocess.call(['amixer', '-D', 'pulse', 'sset', 'Master', f'{percent}%'], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return percent
    except Exception as e:
        print(f"Volume error: {e}")
        return percent

# Volume settings
hmin = 40
hmax = 220
volBar = 400
volPer = get_volume()
initial_volume = volPer
color = (0, 215, 255)

tipIds = [4, 8, 12, 16, 20]
mode = 'N'
active = 0

pyautogui.FAILSAFE = False

# IMPROVED: Faster cursor movement with adaptive smoothening
plocX, plocY = 0, 0
clocX, clocY = 0, 0
smoothening = 2.5  # Reduced from 5 for faster response (lower = faster)

# Enhanced click control
left_click_triggered = False
right_click_triggered = False
last_click_time = 0


def putText(img, text, loc=(250, 450), color=(0, 255, 255), size=3):
    """Display text on image"""
    cv2.putText(img, str(text), loc, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                size, color, 3)


def draw_instructions(img):
    """Draw gesture instructions on screen"""
    instructions = [
        "Gestures:",
        "Fist: Neutral",
        "1 Finger: Scroll Up",
        "2 Fingers: Scroll Down", 
        "Thumb+Index: Volume",
        "  Middle Finger: Exit",
        "All Fingers: Cursor",
        "  T+I Quick: Left Click",
        "  T+I 2x: Double Click",
        "  T+M: Right Click"
    ]
    y_pos = 80
    for i, text in enumerate(instructions):
        cv2.putText(img, text, (10, y_pos + i*22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)


print("=" * 60)
print("Hand Gesture Control - Full Featured Version")
print("=" * 60)
print("\nGestures:")
print("👊 Fist           -> Neutral Mode")
print("☝️  Index finger   -> Scroll Up")
print("✌️  2 Fingers      -> Scroll Down")
print("🤏 Thumb + Index  -> Volume Control")
print("   └─ Middle finger up -> Exit (volume stops)")
print("✋ All 5 fingers  -> Cursor Control (FASTER!)")
print("   ├─ Quick Pinch (T+I)    -> LEFT Click")
print("   ├─ Double Pinch (T+I)   -> DOUBLE Click + Copy")
print("   └─ Pinch (T+M)          -> RIGHT Click")
print("\nPress 'q' to quit")
print("=" * 60)

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Camera read failed!")
            break
        
        # Flip for mirror effect
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        fingers = []

        if len(lmList) != 0:
            # Thumb detection (adjusted for flipped image)
            if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other 4 fingers
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # Mode selection
            if fingers == [0, 0, 0, 0, 0] and active == 0:
                mode = 'N'
            elif (fingers == [0, 1, 0, 0, 0] or fingers == [0, 1, 1, 0, 0]) and active == 0:
                mode = 'Scroll'
                active = 1
            elif fingers == [1, 1, 0, 0, 0] and active == 0:
                mode = 'Volume'
                active = 1
                initial_volume = get_volume()
                volPer = initial_volume
            elif fingers == [1, 1, 1, 1, 1] and active == 0:
                mode = 'Cursor'
                active = 1
                left_click_triggered = False

        ############# SCROLL MODE ##############
        if mode == 'Scroll':
            putText(img, 'SCROLL', (250, 450))
            cv2.rectangle(img, (200, 410), (280, 470), (255, 255, 255), cv2.FILLED)
            
            if len(lmList) != 0:
                if fingers == [0, 1, 0, 0, 0]:
                    putText(img, 'UP', (210, 455), (0, 255, 0), 2)
                    pyautogui.scroll(200)
                    time.sleep(0.2)
                
                elif fingers == [0, 1, 1, 0, 0]:
                    putText(img, 'DN', (210, 455), (0, 0, 255), 2)
                    pyautogui.scroll(-200)
                    time.sleep(0.2)
                
                elif fingers == [0, 0, 0, 0, 0]:
                    active = 0
                    mode = 'N'

        ############# VOLUME MODE ##############
        elif mode == 'Volume':
            putText(img, 'VOLUME', (250, 450))
            
            if len(lmList) != 0:
                # Exit condition: Middle finger raised
                if fingers[2] == 1:
                    active = 0
                    mode = 'N'
                    print(f"Exiting volume mode. Final volume: {int(volPer)}%")
                
                # Check if ONLY thumb and index are up
                elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0:
                    x1, y1 = lmList[4][1], lmList[4][2]
                    x2, y2 = lmList[8][1], lmList[8][2]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    cv2.circle(img, (x1, y1), 12, color, cv2.FILLED)
                    cv2.circle(img, (x2, y2), 12, color, cv2.FILLED)
                    cv2.line(img, (x1, y1), (x2, y2), color, 3)
                    cv2.circle(img, (cx, cy), 10, color, cv2.FILLED)

                    length = math.hypot(x2 - x1, y2 - y1)
                    
                    cv2.putText(img, f'Distance: {int(length)}', (350, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    volPer = np.interp(length, [hmin, hmax], [0, 100])
                    volPer = int(volPer)
                    volPer = max(0, min(100, volPer))
                    
                    set_volume(volPer)
                    volBar = np.interp(volPer, [0, 100], [400, 150])
                    
                    if length < 50:
                        cv2.circle(img, (cx, cy), 13, (0, 0, 255), cv2.FILLED)
                        cv2.putText(img, 'MIN', (cx - 20, cy - 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif length > 200:
                        cv2.circle(img, (cx, cy), 13, (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, 'MAX', (cx - 20, cy - 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.rectangle(img, (30, 150), (55, 400), (0, 255, 0), 3)
                    cv2.rectangle(img, (30, int(volBar)), (55, 400), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, f'{int(volPer)}%', (15, 430), 
                               cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
                    
                    cv2.putText(img, 'Stretch: +  Close: -', (150, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(img, 'Middle Finger Up: Exit', (140, 130), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                else:
                    active = 0
                    mode = 'N'
                    print(f"Fingers changed. Exiting volume mode. Final volume: {int(volPer)}%")

        ############# CURSOR MODE - SIMPLIFIED WITHOUT DRAG ##############
        elif mode == 'Cursor':
            putText(img, 'CURSOR', (250, 450))
            
            # Draw control area
            cv2.rectangle(img, (110, 20), (530, 350), (255, 0, 255), 3)

            if len(lmList) != 0:
                if fingers[1:] == [0, 0, 0, 0]:  # Exit
                    active = 0
                    mode = 'N'
                    left_click_triggered = False
                    right_click_triggered = False
                else:
                    # Get finger positions
                    x_thumb, y_thumb = lmList[4][1], lmList[4][2]
                    x_index, y_index = lmList[8][1], lmList[8][2]
                    x_middle, y_middle = lmList[12][1], lmList[12][2]
                    
                    screen_w, screen_h = pyautogui.size()
                    
                    # IMPROVED: Faster cursor mapping with better range
                    X = np.interp(x_index, [110, 530], [0, screen_w])
                    Y = np.interp(y_index, [20, 350], [0, screen_h])
                    
                    # IMPROVED: Adaptive smoothening - less smooth = faster response
                    clocX = plocX + (X - plocX) / smoothening
                    clocY = plocY + (Y - plocY) / smoothening
                    
                    clocX = max(0, min(clocX, screen_w - 1))
                    clocY = max(0, min(clocY, screen_h - 1))
                    
                    # Visual indicators
                    cv2.circle(img, (x_thumb, y_thumb), 10, (0, 255, 0), cv2.FILLED)
                    cv2.circle(img, (x_index, y_index), 10, (255, 255, 0), cv2.FILLED)
                    cv2.circle(img, (x_middle, y_middle), 10, (255, 0, 255), cv2.FILLED)
                    
                    # Move cursor
                    try:
                        pyautogui.moveTo(clocX, clocY)
                        plocX, plocY = clocX, clocY
                    except:
                        pass
                    
                    # Calculate distances
                    dist_thumb_index = math.hypot(x_index - x_thumb, y_index - y_thumb)
                    dist_thumb_middle = math.hypot(x_middle - x_thumb, y_middle - y_thumb)
                    
                    current_time = time.time()
                    
                    # ========== LEFT CLICK / DOUBLE CLICK ==========
                    if dist_thumb_index < 40 and dist_thumb_middle > 45:
                        if not left_click_triggered:
                            # Check for double click
                            if current_time - last_click_time < 0.5:
                                # DOUBLE CLICK DETECTED
                                cv2.circle(img, (x_thumb, y_thumb), 18, (255, 0, 255), cv2.FILLED)
                                cv2.putText(img, 'DOUBLE CLICK!', (120, 400), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)
                                
                                pyautogui.doubleClick()
                                print("Double click - text selected and copied!")
                                
                                # Copy selected text
                                time.sleep(0.1)
                                pyautogui.hotkey('ctrl', 'c')
                                
                                time.sleep(0.5)
                            else:
                                # SINGLE CLICK
                                cv2.circle(img, (x_thumb, y_thumb), 15, (0, 0, 255), cv2.FILLED)
                                cv2.putText(img, 'CLICK!', (180, 400), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                                pyautogui.click(button='left')
                                print("Single click")
                            
                            last_click_time = current_time
                            left_click_triggered = True
                            time.sleep(0.2)
                    else:
                        left_click_triggered = False
                    
                    # ========== RIGHT CLICK ==========
                    if dist_thumb_middle < 40 and dist_thumb_index > 45:
                        if not right_click_triggered:
                            cv2.circle(img, (x_thumb, y_thumb), 15, (255, 0, 0), cv2.FILLED)
                            cv2.line(img, (x_thumb, y_thumb), (x_middle, y_middle), (255, 0, 0), 3)
                            cv2.putText(img, 'RIGHT CLICK!', (130, 400), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                            
                            pyautogui.click(button='right')
                            right_click_triggered = True
                            print("Right click")
                            time.sleep(0.3)
                    else:
                        right_click_triggered = False
                    
                    # Display distances
                    cv2.putText(img, f'TI:{int(dist_thumb_index)} TM:{int(dist_thumb_middle)}', 
                               (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Display mode and FPS
        cv2.rectangle(img, (0, 0), (200, 60), (50, 50, 50), cv2.FILLED)
        cv2.putText(img, f'Mode: {mode}', (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cTime = time.time()
        fps = 1 / ((cTime + 0.01) - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (450, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        draw_instructions(img)
        cv2.imshow('Hand Gesture Control', img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        
        if cv2.getWindowProperty('Hand Gesture Control', cv2.WND_PROP_VISIBLE) < 1:
            break

except KeyboardInterrupt:
    print("\n\nInterrupted by user")
except Exception as e:
    print(f"\nError occurred: {e}")
finally:
    print("\nCleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print("Exited successfully!")
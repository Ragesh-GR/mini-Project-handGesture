import cv2
import time
import math
import numpy as np
import HandTrackingModule as htm
import pyautogui
import subprocess
import sys
# Ensure console prints Unicode characters properly (Windows)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Configuration
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(maxHands=1, detectionCon=0.85, trackCon=0.8)

# Volume control (cross-platform)
import platform
if platform.system() == 'Windows':
    try:
        from ctypes import POINTER, cast
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        def get_volume():
            try:
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                level = volume.GetMasterVolumeLevelScalar()
                return int(level * 100)
            except:
                return 50
        def set_volume(percent):
            try:
                percent = max(0, min(100, int(percent)))
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                volume.SetMasterVolumeLevelScalar(percent/100.0, None)
                return percent
            except Exception as e:
                print("Volume error: {0}".format(e))
                return percent
    except Exception:
        def get_volume():
            return 50
        def set_volume(percent):
            print('pycaw not available, cannot set volume on Windows')
            return percent
else:
    # Linux amixer fallback
    def get_volume():
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
        try:
            percent = max(0, min(100, int(percent)))
            subprocess.call(['amixer', '-D', 'pulse', 'sset', 'Master', f'{percent}%'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return percent
        except Exception as e:
            print("Volume error: {0}".format(e))
            return percent
# Volume settings
hmin = 50
hmax = 200
volBar = 400
volPer = get_volume()  # Get current volume
prev_volPer = volPer
color = (0, 215, 255)

tipIds = [4, 8, 12, 16, 20]
mode = 'N'
active = 0

pyautogui.FAILSAFE = False

# Smooth cursor movement
plocX, plocY = 0, 0
clocX, clocY = 0, 0
smoothening = 5

# Click control
click_triggered = False
click_cooldown = 0


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
        "All Fingers: Cursor",
        "Pinch in Cursor: Click"
    ]
    y_pos = 80
    for i, text in enumerate(instructions):
        cv2.putText(img, text, (10, y_pos + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


print("=" * 50)
print("Hand Gesture Control - Cross-platform")     
print("=" * 50)
print("\nGestures:")
print("ðŸ‘Š Fist           -> Neutral Mode")
print("â˜ï¸  Index finger   -> Scroll Up")
print("âœŒï¸  2 Fingers      -> Scroll Down")
print("ðŸ¤ Thumb + Index  -> Volume Control")
print("âœ‹ All 5 fingers  -> Cursor Control")
print("   â””â”€ Pinch       -> Click")
print("\nPress 'q' to quit")
print("=" * 50)
# Add a lightweight test runner: run N frames and exit when --test is passed
import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--test', '-t', action='store_true', help='Run in test mode and exit after a few frames')
parser.add_argument('--frames', '-n', type=int, default=60, help='Number of frames to process in test mode')
args, _ = parser.parse_known_args()

try:
    frame_count = 0
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
                prev_volPer = get_volume()  # Get current system volume
            elif fingers == [1, 1, 1, 1, 1] and active == 0:
                mode = 'Cursor'
                active = 1

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
                if fingers[-1] == 1:  # Pinky up to exit
                    active = 0
                    mode = 'N'
                else:
                    x1, y1 = lmList[4][1], lmList[4][2]
                    x2, y2 = lmList[8][1], lmList[8][2]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    cv2.circle(img, (x1, y1), 12, color, cv2.FILLED)
                    cv2.circle(img, (x2, y2), 12, color, cv2.FILLED)
                    cv2.line(img, (x1, y1), (x2, y2), color, 3)
                    cv2.circle(img, (cx, cy), 10, color, cv2.FILLED)

                    length = math.hypot(x2 - x1, y2 - y1)

                    # Smooth volume changes - interpolate from 0-100
                    target_vol = np.interp(length, [hmin, hmax], [0, 100])
                    
                    # Smooth transition - only change if difference > 2%
                    if abs(target_vol - prev_volPer) > 2:
                        volPer = prev_volPer + (target_vol - prev_volPer) * 0.3
                        volPer = max(0, min(100, volPer))
                        set_volume(volPer)
                        prev_volPer = volPer
                    else:
                        volPer = prev_volPer
                    
                    volBar = np.interp(volPer, [0, 100], [400, 150])
                    
                    # Visual feedback
                    if length < 50:
                        cv2.circle(img, (cx, cy), 13, (0, 0, 255), cv2.FILLED)

                    # Volume bar
                    cv2.rectangle(img, (30, 150), (55, 400), (0, 255, 0), 3)
                    cv2.rectangle(img, (30, int(volBar)), (55, 400), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, f'{int(volPer)}%', (15, 430), 
                               cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

        ############# CURSOR MODE ##############
        elif mode == 'Cursor':
            putText(img, 'CURSOR', (250, 450))
            
            # Draw control area
            cv2.rectangle(img, (110, 20), (530, 350), (255, 0, 255), 3)
            cv2.putText(img, 'Move hand here', (200, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            if len(lmList) != 0:
                if fingers[1:] == [0, 0, 0, 0]:  # Exit if only thumb up
                    active = 0
                    mode = 'N'
                    click_triggered = False
                else:
                    # Index finger position for cursor
                    x1, y1 = lmList[8][1], lmList[8][2]
                    
                    # Thumb position for click detection
                    x_thumb, y_thumb = lmList[4][1], lmList[4][2]
                    
                    screen_w, screen_h = pyautogui.size()
                    
                    # Map to screen
                    X = np.interp(x1, [110, 530], [0, screen_w])
                    Y = np.interp(y1, [20, 350], [0, screen_h])
                    
                    # Smooth movement
                    clocX = plocX + (X - plocX) / smoothening
                    clocY = plocY + (Y - plocY) / smoothening
                    
                    clocX = max(0, min(clocX, screen_w - 1))
                    clocY = max(0, min(clocY, screen_h - 1))
                    
                    # Visual indicators
                    cv2.circle(img, (lmList[8][1], lmList[8][2]), 10, (255, 255, 0), cv2.FILLED)
                    cv2.circle(img, (x_thumb, y_thumb), 10, (0, 255, 0), cv2.FILLED)
                    
                    # Move cursor
                    try:
                        pyautogui.moveTo(clocX, clocY)
                        plocX, plocY = clocX, clocY
                    except:
                        pass
                    
                    # Click detection - measure distance between thumb and index
                    distance = math.hypot(x1 - x_thumb, y1 - y_thumb)
                    
                    # Click when pinched (distance < 40)
                    if distance < 40:
                        if not click_triggered:
                            cv2.circle(img, (x_thumb, y_thumb), 15, (0, 0, 255), cv2.FILLED)
                            cv2.putText(img, 'CLICK!', (200, 400), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                            pyautogui.click()
                            click_triggered = True
                            time.sleep(0.3)
                    else:
                        click_triggered = False

        # Display mode and FPS
        cv2.rectangle(img, (0, 0), (200, 60), (50, 50, 50), cv2.FILLED)
        cv2.putText(img, f'Mode: {mode}', (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cTime = time.time()
        fps = 1 / ((cTime + 0.01) - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (450, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw instructions
        draw_instructions(img)

        cv2.imshow('Hand Gesture Control', img)

        # Increment frame count and optionally exit for test mode
        frame_count += 1
        if args.test and frame_count >= args.frames:
            print(f"Test mode: processed {frame_count} frames, exiting")
            break   

        # Increment frame count and optionally exit for test mode
        frame_count += 1
        if args.test and frame_count >= args.frames:
            print(f"Test mode: processed {frame_count} frames, exiting")
            break

        # Proper exit handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        
        # Check if window was closed
        if cv2.getWindowProperty('Hand Gesture Control', cv2.WND_PROP_VISIBLE) < 1:
            break

except KeyboardInterrupt:
    print("\n\nInterrupted by user")
except Exception as e:
    print(f"\nError occurred: {e}")
finally:
    # Cleanup
    print("\nCleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Extra wait to ensure window closes
    print("Exited successfully!")

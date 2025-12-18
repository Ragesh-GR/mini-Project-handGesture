import cv2
import csv
import os
import numpy as np
import HandTrackingModule as htm
from datetime import datetime

class GestureDataCollector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        self.detector = htm.handDetector(maxHands=1, detectionCon=0.8)
        
        # Complete alphabet + space, backspace, enter
        self.gestures = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['SPACE', 'BACK', 'ENTER']
        self.current_index = 0
        self.current_gesture = self.gestures[self.current_index]
        
        # Create dataset directory
        os.makedirs('gesture_dataset', exist_ok=True)
        self.dataset_file = 'gesture_dataset/gestures.csv'
        
        # Initialize CSV if needed
        if not os.path.exists(self.dataset_file):
            with open(self.dataset_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header: 21 landmarks * 3 coords (x,y,z) + label
                header = []
                for i in range(21):
                    header.extend([f'x{i}', f'y{i}', f'z{i}'])
                header.append('label')
                writer.writerow(header)
        
        self.file = open(self.dataset_file, 'a', newline='')
        self.writer = csv.writer(self.file)
        self.sample_count = 0
        self.recording = False
        
    def normalize_landmarks(self, lmList):
        """Normalize landmarks relative to wrist position"""
        if len(lmList) == 0:
            return None
        
        # Extract coordinates
        coords = np.array([[lm[1], lm[2], lm[3]] for lm in lmList])
        
        # Normalize relative to wrist (landmark 0)
        wrist = coords[0]
        normalized = coords - wrist
        
        # Scale to unit variance
        max_dist = np.max(np.abs(normalized))
        if max_dist > 0:
            normalized = normalized / max_dist
        
        return normalized.flatten().tolist()
    
    def draw_ui(self, img):
        """Draw collection interface"""
        # Top bar - current gesture
        cv2.rectangle(img, (0, 0), (640, 80), (50, 50, 50), cv2.FILLED)
        cv2.putText(img, f'Gesture: {self.current_gesture}', (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        # Sample counter
        cv2.rectangle(img, (0, 400), (200, 480), (50, 50, 50), cv2.FILLED)
        cv2.putText(img, f'Samples: {self.sample_count}', (10, 440), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Recording indicator
        if self.recording:
            cv2.circle(img, (600, 450), 20, (0, 0, 255), cv2.FILLED)
            cv2.putText(img, 'REC', (550, 470), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Instructions
        instructions = [
            "SPACE: Start/Stop Recording",
            "N: Next Gesture",
            "P: Previous Gesture", 
            "Q: Quit",
            "",
            f"Progress: {self.current_index + 1}/{len(self.gestures)}"
        ]
        
        y_pos = 100
        for instr in instructions:
            cv2.putText(img, instr, (420, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 25
    
    def run(self):
        print("=" * 60)
        print("VIRTUAL KEYBOARD - GESTURE DATA COLLECTION")
        print("=" * 60)
        print("\nControls:")
        print("  SPACE - Start/Stop Recording")
        print("  N     - Next Gesture")
        print("  P     - Previous Gesture")
        print("  Q     - Quit")
        print("\nCollect at least 100 samples per gesture for best results")
        print("=" * 60)
        
        try:
            while True:
                success, img = self.cap.read()
                if not success:
                    break
                
                img = cv2.flip(img, 1)
                img = self.detector.findHands(img)
                lmList = self.detector.findPosition(img, draw=True, z_axis=True)
                
                # Record sample if recording and hand detected
                if self.recording and len(lmList) != 0:
                    normalized = self.normalize_landmarks(lmList)
                    if normalized:
                        row = normalized + [self.current_gesture]
                        self.writer.writerow(row)
                        self.sample_count += 1
                
                self.draw_ui(img)
                cv2.imshow('Gesture Data Collection', img)
                
                key = cv2.waitKey(1) & 0xFF
                
                # Toggle recording
                if key == ord(' '):
                    self.recording = not self.recording
                    if self.recording:
                        print(f"Recording {self.current_gesture}...")
                    else:
                        print(f"Stopped. Collected {self.sample_count} samples")
                
                # Next gesture
                elif key == ord('n'):
                    self.current_index = (self.current_index + 1) % len(self.gestures)
                    self.current_gesture = self.gestures[self.current_index]
                    self.sample_count = 0
                    self.recording = False
                    print(f"\nSwitched to: {self.current_gesture}")
                
                # Previous gesture
                elif key == ord('p'):
                    self.current_index = (self.current_index - 1) % len(self.gestures)
                    self.current_gesture = self.gestures[self.current_index]
                    self.sample_count = 0
                    self.recording = False
                    print(f"\nSwitched to: {self.current_gesture}")
                
                # Quit
                elif key == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        finally:
            self.cleanup()
        self.cap.release()
        cv2.destroyAllWindows()

    def cleanup(self):
        print("\nSaving and closing...")
        self.file.close()
        self.cap.release()
        cv2.destroyAllWindows()
        print("Done!")

if __name__ == "__main__":
    collector = GestureDataCollector()
    collector.run()
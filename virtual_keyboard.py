import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import HandTrackingModule as htm
import time
from collections import deque

class VirtualKeyboard:
    def __init__(self, model_path='models/gesture_model.h5', 
                 encoder_path='models/label_encoder.pkl'):
        # Load model and encoder
        print("Loading model...")
        self.model = keras.models.load_model(model_path)
        self.label_encoder = joblib.load(encoder_path)
        print("Model loaded successfully!")
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        
        # Hand detector
        self.detector = htm.handDetector(maxHands=1, detectionCon=0.85, trackCon=0.8)
        
        # Text state
        self.typed_text = ""
        self.max_text_length = 500
        
        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=5)
        self.last_prediction = None
        self.prediction_confidence = 0.0
        self.min_confidence = 0.75
        
        # Gesture timing
        self.last_gesture_time = 0
        self.gesture_cooldown = 1.0  # seconds between gestures
        self.hold_time = 0.8  # time to hold gesture before registration
        self.gesture_start_time = None
        self.holding_gesture = None
        
        # UI state
        self.show_keyboard = True
        self.keyboard_opacity = 0.7
        
        # Virtual keyboard layout
        self.keyboard_layout = [
            list('QWERTYUIOP'),
            list('ASDFGHJKL'),
            list('ZXCVBNM'),
            ['SPACE', 'BACK', 'ENTER']
        ]
        
    def normalize_landmarks(self, lmList):
        """Normalize landmarks for model input"""
        if len(lmList) == 0:
            return None
        
        coords = np.array([[lm[1], lm[2], lm[3]] for lm in lmList])
        wrist = coords[0]
        normalized = coords - wrist
        
        max_dist = np.max(np.abs(normalized))
        if max_dist > 0:
            normalized = normalized / max_dist
        
        return normalized.flatten()
    
    def predict_gesture(self, lmList):
        """Predict gesture from landmarks"""
        if len(lmList) == 0:
            return None, 0.0
        
        normalized = self.normalize_landmarks(lmList)
        if normalized is None:
            return None, 0.0
        
        # Predict
        input_data = normalized.reshape(1, -1)
        prediction = self.model.predict(input_data, verbose=0)
        
        # Get class and confidence
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        gesture = self.label_encoder.inverse_transform([class_idx])[0]
        
        return gesture, confidence
    
    def smooth_prediction(self, gesture, confidence):
        """Smooth predictions using buffer"""
        if confidence < self.min_confidence:
            return None, 0.0
        
        self.prediction_buffer.append(gesture)
        
        # Most common gesture in buffer
        if len(self.prediction_buffer) >= 3:
            from collections import Counter
            counts = Counter(self.prediction_buffer)
            most_common = counts.most_common(1)[0]
            if most_common[1] >= 3:  # At least 3 same predictions
                return most_common[0], confidence
        
        return None, 0.0
    
    def process_gesture(self, gesture):
        """Process confirmed gesture"""
        current_time = time.time()
        
        # Check if enough time has passed
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return False
        
        # Process special gestures
        if gesture == 'SPACE':
            self.typed_text += ' '
        elif gesture == 'BACK':
            self.typed_text = self.typed_text[:-1]
        elif gesture == 'ENTER':
            self.typed_text += '\n'
        else:
            # Regular letter
            if len(self.typed_text) < self.max_text_length:
                self.typed_text += gesture
        
        self.last_gesture_time = current_time
        return True
    
    def draw_virtual_keyboard(self, img):
        """Draw virtual keyboard overlay"""
        if not self.show_keyboard:
            return
        
        # Keyboard dimensions
        kb_x, kb_y = 50, 500
        key_width, key_height = 70, 60
        key_spacing = 10
        
        # Semi-transparent overlay
        overlay = img.copy()
        
        for row_idx, row in enumerate(self.keyboard_layout):
            x_offset = kb_x + (row_idx * 25)  # Indent each row
            
            for col_idx, key in enumerate(row):
                x = x_offset + col_idx * (key_width + key_spacing)
                y = kb_y + row_idx * (key_height + key_spacing)
                
                # Special keys are wider
                width = key_width * 2 if key in ['SPACE', 'BACK', 'ENTER'] else key_width
                
                # Highlight if current prediction
                color = (0, 255, 0) if key == self.holding_gesture else (100, 100, 100)
                
                cv2.rectangle(overlay, (x, y), (x + width, y + key_height), 
                            color, -1)
                cv2.rectangle(overlay, (x, y), (x + width, y + key_height), 
                            (255, 255, 255), 2)
                
                # Draw key text
                text_size = 0.7 if len(key) == 1 else 0.5
                text_thickness = 2
                text = key
                
                text_w, text_h = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                 text_size, text_thickness)[0]
                text_x = x + (width - text_w) // 2
                text_y = y + (key_height + text_h) // 2
                
                cv2.putText(overlay, text, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 255), 
                          text_thickness)
        
        # Blend overlay
        cv2.addWeighted(overlay, self.keyboard_opacity, img, 1 - self.keyboard_opacity, 0, img)
    
    def draw_text_area(self, img):
        """Draw text input area"""
        # Text box
        cv2.rectangle(img, (50, 50), (1230, 200), (50, 50, 50), -1)
        cv2.rectangle(img, (50, 50), (1230, 200), (255, 255, 255), 2)
        
        # Title
        cv2.putText(img, "Virtual Keyboard Input:", (60, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Typed text with word wrapping
        words = self.typed_text.split(' ')
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + word + " "
            if len(test_line) > 60:  # Max characters per line
                lines.append(current_line)
                current_line = word + " "
            else:
                current_line = test_line
        
        if current_line:
            lines.append(current_line)
        
        # Display lines (max 3)
        y_pos = 120
        for line in lines[-3:]:
            cv2.putText(img, line, (60, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_pos += 35
        
        # Character count
        cv2.putText(img, f"Characters: {len(self.typed_text)}/{self.max_text_length}", 
                   (60, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def draw_prediction_info(self, img, gesture, confidence):
        """Draw current prediction info"""
        # Prediction box
        cv2.rectangle(img, (50, 230), (400, 350), (50, 50, 50), -1)
        cv2.rectangle(img, (50, 230), (400, 350), (255, 255, 255), 2)
        
        cv2.putText(img, "Current Gesture:", (60, 260),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if gesture:
            # Gesture name
            cv2.putText(img, gesture, (60, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # Confidence bar
            cv2.putText(img, f"Confidence: {confidence*100:.1f}%", (60, 330),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            bar_width = int(300 * confidence)
            cv2.rectangle(img, (60, 335), (60 + bar_width, 345), (0, 255, 0), -1)
            cv2.rectangle(img, (60, 335), (360, 345), (255, 255, 255), 1)
        else:
            cv2.putText(img, "No gesture", (60, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
    
    def draw_hold_indicator(self, img):
        """Draw hold progress indicator"""
        if self.holding_gesture and self.gesture_start_time:
            elapsed = time.time() - self.gesture_start_time
            progress = min(elapsed / self.hold_time, 1.0)
            
            # Progress circle
            center = (1150, 300)
            radius = 40
            
            # Background
            cv2.circle(img, center, radius, (50, 50, 50), -1)
            cv2.circle(img, center, radius, (255, 255, 255), 2)
            
            # Progress arc
            angle = int(360 * progress)
            if angle > 0:
                cv2.ellipse(img, center, (radius-5, radius-5), -90, 0, angle, 
                          (0, 255, 0), 8)
            
            # Text
            cv2.putText(img, "HOLD", (center[0]-30, center[1]+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_instructions(self, img):
        """Draw control instructions"""
        instructions = [
            "Controls:",
            "K - Toggle Keyboard",
            "C - Clear Text",
            "S - Save Text",
            "Q - Quit",
            "",
            "Hold gesture for 0.8s to type"
        ]
        
        y_pos = 380
        for instr in instructions:
            cv2.putText(img, instr, (60, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_pos += 25
    
    def save_text(self):
        """Save typed text to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"typed_text_{timestamp}.txt"
        with open(filename, 'w') as f:
            f.write(self.typed_text)
        print(f"Text saved to {filename}")
        return filename
    
    def run(self):
        """Main application loop"""
        print("=" * 60)
        print("VIRTUAL KEYBOARD - GESTURE TYPING")
        print("=" * 60)
        print("\nInstructions:")
        print("  1. Show a gesture to the camera")
        print("  2. Hold the gesture for 0.8 seconds")
        print("  3. The letter will be typed")
        print("\nControls:")
        print("  K - Toggle virtual keyboard display")
        print("  C - Clear all text")
        print("  S - Save text to file")
        print("  Q - Quit")
        print("=" * 60)
        
        try:
            while True:
                success, img = self.cap.read()
                if not success:
                    break
                
                img = cv2.flip(img, 1)
                img = self.detector.findHands(img, draw=True)
                lmList = self.detector.findPosition(img, draw=False, z_axis=True)
                
                # Predict gesture
                gesture, confidence = self.predict_gesture(lmList)
                smoothed_gesture, smoothed_conf = self.smooth_prediction(gesture, confidence)
                
                # Handle gesture hold timing
                current_time = time.time()
                
                if smoothed_gesture and smoothed_conf > self.min_confidence:
                    if self.holding_gesture != smoothed_gesture:
                        # New gesture detected
                        self.holding_gesture = smoothed_gesture
                        self.gesture_start_time = current_time
                    else:
                        # Same gesture - check hold time
                        hold_duration = current_time - self.gesture_start_time
                        if hold_duration >= self.hold_time:
                            if self.process_gesture(smoothed_gesture):
                                # Reset after successful input
                                self.holding_gesture = None
                                self.gesture_start_time = None
                else:
                    # No valid gesture
                    self.holding_gesture = None
                    self.gesture_start_time = None
                
                # Draw UI
                self.draw_text_area(img)
                self.draw_prediction_info(img, smoothed_gesture, smoothed_conf)
                self.draw_virtual_keyboard(img)
                self.draw_hold_indicator(img)
                self.draw_instructions(img)
                
                # FPS
                cv2.putText(img, f"FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))}", 
                           (1150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Virtual Keyboard', img)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('k'):
                    self.show_keyboard = not self.show_keyboard
                elif key == ord('c'):
                    self.typed_text = ""
                    print("Text cleared")
                elif key == ord('s'):
                    filename = self.save_text()
                    print(f"Saved to {filename}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        print("Exited successfully!")

if __name__ == "__main__":
    keyboard = VirtualKeyboard()
    keyboard.run()
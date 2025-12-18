# 🖐️ AI-Powered Hand Gesture Control System

A comprehensive hand gesture recognition system that enables touchless computer control and virtual keyboard typing using computer vision and deep learning.

## 🌟 Features

### 1. **Hand Gesture Control** (`main.py`)
Control your computer without touching anything!

- 👊 **Neutral Mode** - Fist (reset to neutral)
- 📜 **Scroll Mode** - One finger up (scroll up), two fingers (scroll down)
- 🔊 **Volume Control** - Thumb + Index finger distance controls volume
- 🖱️ **Cursor Control** - All 5 fingers up to move cursor
  - **Left Click** - Quick pinch (Thumb + Index)
  - **Double Click** - Two quick pinches + auto copy text
  - **Right Click** - Pinch (Thumb + Middle finger)

### 2. **Virtual Keyboard** (`virtual_keyboard.py`)
Type text using hand gestures with AI recognition!

- 26 alphabet letters (A-Z)
- Special commands: SPACE, BACKSPACE, ENTER
- Real-time gesture prediction with confidence display
- Auto-save typed text to file
- Smart gesture hold detection (0.8s hold to type)

### 3. **Data Collection & Training**
Build your own gesture recognition model!

- `collect_data.py` - Collect training data for gestures
- `train_model.py` - Train CNN model for gesture classification

---

## 📋 Requirements

### System Requirements
- **OS**: Linux (Ubuntu recommended), Windows, or macOS
- **Python**: 3.8 or higher
- **Webcam**: Required for hand detection
- **RAM**: 4GB minimum (8GB recommended)

### Python Dependencies
Install all required packages:

```bash
pip install -r requirements.txt
```

**Required Libraries:**
- opencv-python==4.8.1.78
- mediapipe==0.10.8
- numpy==1.24.3
- pyautogui==0.9.54
- tensorflow==2.15.0
- scikit-learn==1.3.2
- joblib==1.3.2
- pandas==2.0.3
- matplotlib==3.7.3

---

## 🚀 Quick Start

### Option 1: Hand Gesture Control (Ready to Use)

```bash
# Run the gesture control system
python main.py
```

**Controls:**
- Show gestures to the camera
- Press `Q` to quit

### Option 2: Virtual Keyboard (Requires Training)

#### Step 1: Collect Training Data
```bash
python collect_data.py
```

**Instructions:**
- Press `SPACE` to start/stop recording
- Press `N` for next gesture, `P` for previous
- Collect at least **100 samples per gesture**
- Press `Q` to quit

**Tip:** Make varied hand poses (different angles, distances, lighting) for better accuracy.

#### Step 2: Train the Model
```bash
python train_model.py
```

This will:
- Load collected data from `gesture_dataset/gestures.csv`
- Train a CNN model
- Save model to `models/gesture_model.h5`
- Save label encoder to `models/label_encoder.pkl`
- Generate training history plot

**Expected Training Time:** 5-15 minutes (depending on dataset size and hardware)

#### Step 3: Use Virtual Keyboard
```bash
python virtual_keyboard.py
```

**Controls:**
- Hold gesture for 0.8 seconds to type
- `K` - Toggle keyboard display
- `C` - Clear text
- `S` - Save text to file
- `Q` - Quit

---

## 📁 Project Structure

```
hand-gesture-control/
│
├── main.py                      # Gesture control application
├── virtual_keyboard.py          # Virtual keyboard application
├── collect_data.py              # Data collection tool
├── train_model.py               # Model training script
├── HandTrackingModule.py        # Hand detection module
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── gesture_dataset/             # Training data (auto-created)
│   └── gestures.csv
│
├── models/                      # Trained models (auto-created)
│   ├── gesture_model.h5
│   ├── label_encoder.pkl
│   └── training_history.png
│
└── typed_text_YYYYMMDD_HHMMSS.txt  # Saved keyboard outputs
```

---

## 🎯 Gesture Guide

### Hand Gesture Control Modes

| Gesture | Fingers | Action |
|---------|---------|--------|
| 👊 Fist | All down | Neutral mode |
| ☝️ One finger | Index up | Scroll up |
| ✌️ Two fingers | Index + Middle up | Scroll down |
| 🤏 Pinch (T+I) | Thumb + Index | Volume control / Click |
| 🖖 All fingers | All 5 up | Cursor control mode |

### Cursor Mode Controls

| Action | Gesture | Description |
|--------|---------|-------------|
| Move | All fingers up | Move cursor with index finger |
| Click | Quick pinch (T+I) | Single left click |
| Double Click | Double pinch (T+I) | Double click + copy |
| Right Click | Pinch (T+M) | Right click menu |

### Virtual Keyboard Gestures

Show the gesture corresponding to the letter you want to type:
- **A-Z**: Individual hand shapes for each letter
- **SPACE**: Specific gesture for space
- **BACK**: Backspace gesture
- **ENTER**: Enter/newline gesture

---

## ⚙️ Configuration

### Volume Control (Linux)

The system uses `amixer` for volume control on Linux. For other systems:

**Windows:**
Replace volume functions in `main.py` with:
```python
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
```

**macOS:**
Use `osascript` commands:
```python
import subprocess
subprocess.call(['osascript', '-e', f'set volume output volume {percent}'])
```

### Adjusting Sensitivity

In `main.py`, modify these parameters:

```python
# Detection confidence (0.0 to 1.0)
detector = htm.handDetector(maxHands=1, detectionCon=0.85, trackCon=0.8)

# Cursor smoothness (lower = faster, higher = smoother)
smoothening = 2.5

# Click distance threshold
dist_thumb_index < 40  # Decrease for tighter pinch
```

In `virtual_keyboard.py`:

```python
# Prediction confidence threshold
self.min_confidence = 0.75  # Increase for more accuracy, decrease for responsiveness

# Hold time before typing
self.hold_time = 0.8  # seconds

# Cooldown between gestures
self.gesture_cooldown = 1.0  # seconds
```

---

## 🐛 Troubleshooting

### Camera Not Working
```bash
# Check available cameras
ls /dev/video*

# Change camera index in code
cap = cv2.VideoCapture(1)  # Try 0, 1, 2, etc.
```

### Hand Not Detected
- Ensure good lighting
- Keep hand within camera frame
- Increase detection confidence:
  ```python
  detector = htm.handDetector(detectionCon=0.7)  # Lower value
  ```

### Low FPS
- Close other applications
- Reduce camera resolution:
  ```python
  cap.set(3, 640)  # Width
  cap.set(4, 480)  # Height
  ```

### Model Not Loading
```bash
# Check if model files exist
ls models/

# If missing, retrain:
python train_model.py
```

### Volume Control Not Working (Linux)
```bash
# Install ALSA utilities
sudo apt-get install alsa-utils

# Test volume command
amixer set Master 50%
```

---

## 📊 Training Tips

### For Best Model Accuracy:

1. **Diverse Data Collection**
   - Collect 150-200 samples per gesture
   - Vary hand positions, angles, distances
   - Include different lighting conditions
   - Use both left and right hands

2. **Consistent Gestures**
   - Make clear, distinct hand shapes
   - Hold gesture steady during recording
   - Avoid ambiguous poses

3. **Data Quality**
   - Remove corrupted samples from CSV
   - Balance samples across all gestures
   - Check for duplicate gestures

4. **Model Training**
   - Train for 100+ epochs
   - Monitor validation accuracy
   - Early stopping prevents overfitting
   - Aim for 90%+ test accuracy

---

## 🎓 How It Works

### Hand Tracking
Uses **MediaPipe Hands** to detect 21 hand landmarks in real-time:
- Wrist, thumb, index, middle, ring, pinky joints
- 3D coordinates (x, y, z) for each landmark

### Gesture Recognition
1. **Capture** hand landmarks from camera
2. **Normalize** coordinates relative to wrist
3. **Extract** features (finger positions, distances)
4. **Classify** gesture using CNN model
5. **Execute** corresponding action

### CNN Model Architecture
```
Input (63 features) → Reshape (21 landmarks × 3 coords)
    ↓
Conv1D(64) → BatchNorm → MaxPool → Dropout(0.3)
    ↓
Conv1D(128) → BatchNorm → MaxPool → Dropout(0.3)
    ↓
Conv1D(256) → BatchNorm → GlobalAvgPool
    ↓
Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Dense(128) → BatchNorm → Dropout(0.4)
    ↓
Dense(64) → Dropout(0.3)
    ↓
Output (num_classes) → Softmax
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Ideas for Contribution:
- Add more gestures
- Improve model accuracy
- Cross-platform volume control
- Mobile app integration
- Voice feedback
- Multi-hand support

---

## 📝 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- **MediaPipe** - Hand tracking solution
- **TensorFlow** - Deep learning framework
- **OpenCV** - Computer vision library
- **PyAutoGUI** - GUI automation

---

## 📧 Contact & Support

Found a bug? Have a suggestion? 

- **Issues**: Open an issue on GitHub
- **Discussions**: Start a discussion for questions
- **Pull Requests**: Contributions welcome!

---

## 🎉 Demo Videos

*(Add links to demo videos or GIFs here)*

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Made with ❤️ using Python, OpenCV, and TensorFlow**

*Last Updated: December 2024*

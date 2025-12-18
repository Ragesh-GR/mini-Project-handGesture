# Hand Gesture Control (patched)

Cross-platform hand gesture control using MediaPipe + OpenCV.

Quick start (Windows, recommended with Python 3.11):

```powershell
# from repo root
python -m venv venv311
.\venv311\Scripts\python -m pip install --upgrade pip setuptools wheel
.\venv311\Scripts\python -m pip install -r requirements.txt
.\venv311\Scripts\python main.py
```

Test mode (no interactive quit required):

```powershell
.\venv311\Scripts\python main.py --test --frames 120
```

Notes:
- Windows: volume control uses `pycaw` (added); if `pycaw` isn't available the app will fall back gracefully.
- If pip tries to build numpy from source, ensure you are running Python 3.11 (we installed Python 3.11 in the workspace).
- The script uses the webcam and `pyautogui` to move/click the mouse — be cautious when it runs (you can disable `pyautogui.FAILSAFE`).

If you want, I can open a pull request with these changes and a short description of what I did (Windows support, test mode, README).
# Face Filters (PySide6 + OpenCV)

Real-time webcam face filters with sunglasses, hat, mustache, halo, sparkles, bow tie, and face-frame overlays. Uses Haar cascades for quick detection and MediaPipe Face Mesh for accurate landmark placement.

## Requirements

- Python 3.11
- Webcam

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Or use the helper script:

```powershell
scripts\create_venv.ps1
```

## Run

```powershell
.\.venv\Scripts\Activate.ps1
python -m face_graphics
```

## Configuration

Set custom cascade paths if you want to override OpenCV defaults:

```powershell
$env:FACE_CASCADE_PATH = "C:\\path\\to\\haarcascade_frontalface_alt.xml"
$env:EYE_CASCADE_PATH = "C:\\path\\to\\haarcascade_eye_tree_eyeglasses.xml"
```

The MediaPipe landmarker model is downloaded automatically on first run to:

```
models/face_landmarker.task
```

If you want a custom location, set:

```powershell
$env:FACE_LANDMARKER_PATH = "C:\\path\\to\\face_landmarker.task"
```

Hand gesture rotation uses a separate model downloaded automatically to:

```
models/hand_landmarker.task
```

Override the path if needed:

```powershell
$env:HAND_LANDMARKER_PATH = "C:\\path\\to\\hand_landmarker.task"
```

## Controls

- Toggle individual graphic objects: sunglasses, hat, mustache, halo, sparkles, bow tie, and face frame.
- Adjust opacity, scale, shadow, style, tint color, and accent color.
- Choose a rotation target so manual rotation, auto motion, position offsets, and hand gestures affect all objects or one selected object.
- Tune rotation with Sway, Spin, and Pulse motion modes plus speed and amplitude controls.
- Use target X/Y offsets to nudge a selected graphic into place.
- Apply camera effects such as soft, cinematic, cartoon, edges, and grayscale.
- Use Clean, Party, and Cyber presets for quick looks.
- Enable hand pinch rotation, then pinch your thumb and index finger to grab and twist the selected target without snapping. Use hand sensitivity to control how strongly your wrist motion rotates the graphic. Open your hand to slightly scale the same target.
- Use Snapshot to save the current rendered frame.

# Face Filter

Real-time webcam face filters with sunglasses, hat, mustache, halo, sparkles, bow tie, and face-frame overlays. Uses Haar cascades for quick detection and MediaPipe Face Mesh for accurate landmark placement.

## Requirements

- Python 3.11 or 3.12
- A webcam
- Internet access on first run if the MediaPipe model files are not already in `models/`
- Windows PowerShell for the commands below

## Fresh Setup

```powershell
git clone https://github.com/GihanIllukkumbura/face_filter.git
cd face_filter

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

You can also use the helper script after cloning:

```powershell
.\scripts\create_venv.ps1
.\.venv\Scripts\Activate.ps1
```

## Run

```powershell
.\.venv\Scripts\Activate.ps1
python -m face_graphics
```

The app opens a desktop window and uses your default webcam. The first run may take longer because the Face Landmarker and Hand Landmarker models are downloaded automatically.

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

To choose a different webcam, edit `camera_index` in `face_graphics/config.py`.

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

## Troubleshooting

- If `python` is not recognized, install Python from python.org and enable the "Add python.exe to PATH" option.
- If the camera does not open, close other apps using the webcam and try again.
- If model download fails, check your internet connection, then rerun `python -m face_graphics`.
- If dependency installation fails, upgrade pip first:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

# Laptop Setup Guide (Windows)

Quick setup guide for running the fall detection system on a Windows laptop with webcam.

---

## Prerequisites

- Python 3.9+ installed
- Webcam (built-in or USB)

---

## Setup Steps

### 1. Open PowerShell

Navigate to the project folder:

```powershell
cd c:\MyStuff\CodingProjects\VisioNull
```

### 2. Create Virtual Environment (First Time Only)

```powershell
python -m venv venv
```

### 3. Activate Virtual Environment

**Run this every time you open a new terminal:**

```powershell
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` at the start of your prompt:
```
(venv) PS C:\MyStuff\CodingProjects\VisioNull>
```

> **Note:** If you get an execution policy error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 4. Install Dependencies (First Time Only)

```powershell
pip install -r requirements.txt
```

---

## Running the Application

**Always ensure `(venv)` is shown in your prompt first!**

### Test Camera

```powershell
python -m src.camera_stream
```

Press `Q` to quit.

### Test Pose Detection

```powershell
python -m src.pose_estimator
```

Stand in front of webcam - you should see skeleton overlay.

### Run Fall Detection

```powershell
python -m src.main
```

### Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `D` | Toggle debug info |
| `R` | Reset detector |

---

## Troubleshooting

### "No module named 'cv2'" Error

You forgot to activate the virtual environment. Run:

```powershell
.\venv\Scripts\Activate.ps1
```

### Camera Not Found

Try a different camera index:

```powershell
python -m src.main --camera 1
```

### Execution Policy Error

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Deactivate When Done

```powershell
deactivate
```

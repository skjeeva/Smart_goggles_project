# Smart Goggles for Visually Impaired — Complete Setup & Deployment Guide

> **Target hardware:** Raspberry Pi 4 Model B (4 GB RAM)  
> **OS:** Raspberry Pi OS (Bookworm, 64-bit recommended)  
> **Python:** 3.9 – 3.11

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Hardware Requirements](#2-hardware-requirements)
3. [OS Preparation](#3-os-preparation)
4. [System Package Installation](#4-system-package-installation)
5. [Python Environment Setup](#5-python-environment-setup)
6. [Python Package Installation](#6-python-package-installation)
7. [Camera Configuration](#7-camera-configuration)
8. [Audio & Microphone Configuration](#8-audio--microphone-configuration)
9. [Vosk Speech Model Setup](#9-vosk-speech-model-setup)
10. [API Key Configuration](#10-api-key-configuration)
11. [Running the Application](#11-running-the-application)
12. [Auto-Start on Boot (Production)](#12-auto-start-on-boot-production)
13. [Configuration Tuning](#13-configuration-tuning)
14. [Troubleshooting](#14-troubleshooting)
15. [Architecture Reference](#15-architecture-reference)

---

## 1. System Overview

The Smart Goggles system provides two capabilities running simultaneously:

| Feature | Thread | Purpose |
|---------|--------|---------|
| **YOLO Obstacle Detection** | Thread 1 (continuous) | Detects objects using YOLOv8-nano, estimates distance via monocular depth, announces hazards with clock-position directions |
| **Vision Q&A Assistant** | Thread 2 (wake-word triggered) | Listens for "hello", records a spoken question, sends camera frame + question to a vision LLM, speaks the answer |

Both threads share a single camera through a thread-safe wrapper and push speech into a priority queue consumed by a dedicated TTS thread.

---

## 2. Hardware Requirements

| Component | Specification | Notes |
|-----------|--------------|-------|
| Raspberry Pi | 4 Model B, **4 GB RAM** | 8 GB works too; 2 GB may struggle |
| MicroSD card | 32 GB+ Class 10 / U1 | A2-rated recommended |
| Camera | USB webcam **or** RPi Camera Module v2/v3 | USB is easier to set up |
| Microphone | USB mic or USB audio adapter + mic | Required for wake-word |
| Speaker / Earphone | 3.5 mm jack, Bluetooth, or USB speaker | Required for TTS output |
| Power supply | 5V 3A USB-C | Official RPi PSU recommended |
| Internet | Wi-Fi or Ethernet | Required for LLM API calls |
| (Optional) | Battery pack, 3D-printed goggle frame | For wearable deployment |

---

## 3. OS Preparation

### 3.1 Flash Raspberry Pi OS

1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. Select **Raspberry Pi OS (64-bit)** — Bookworm
3. In Settings (⚙), configure:
   - Hostname: `smartgoggles`
   - Enable SSH
   - Set Wi-Fi SSID and password
   - Set username/password (e.g., `pi` / your password)
4. Flash to microSD, boot the Pi

### 3.2 Initial system update

```bash
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

### 3.3 Enable camera interface (RPi Camera Module only)

```bash
# For RPi Camera Module v2/v3:
sudo raspi-config
# → Interface Options → Camera → Enable
# → reboot
```

For USB webcams, no raspi-config change is needed.

### 3.4 Set GPU memory split

```bash
# Give more RAM to the CPU (YOLO needs it)
echo "gpu_mem=128" | sudo tee -a /boot/firmware/config.txt
sudo reboot
```

---

## 4. System Package Installation

Run all commands as your normal user (with sudo where shown):

```bash
# Build tools & core libraries
sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    wget \
    curl \
    unzip

# Python development headers
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-venv

# OpenCV system dependencies
sudo apt install -y \
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \
    libhdf5-dev \
    libharfbuzz-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libv4l-dev \
    v4l-utils

# Audio system dependencies
sudo apt install -y \
    espeak \
    espeak-ng \
    libespeak-dev \
    portaudio19-dev \
    libasound2-dev \
    pulseaudio \
    alsa-utils \
    libsndfile1-dev

# Useful debugging tools
sudo apt install -y \
    htop \
    lsusb \
    usbutils
```

**Verify camera device exists:**

```bash
ls -la /dev/video*
# Should show /dev/video0 (and possibly video1, video2 …)

v4l2-ctl --list-devices
# Should list your USB camera or RPi camera
```

**Verify audio works:**

```bash
# Test speaker
espeak "Hello, testing audio output"

# List audio devices
aplay -l    # playback devices
arecord -l  # recording devices (microphone)
```

---

## 5. Python Environment Setup

Use a virtual environment to avoid conflicts with system packages:

```bash
# Create virtual environment
python3 -m venv ~/goggles-env

# Activate it (add to .bashrc for convenience)
source ~/goggles-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

**Add to `.bashrc` for auto-activation:**

```bash
echo 'source ~/goggles-env/bin/activate' >> ~/.bashrc
```

---

## 6. Python Package Installation

### 6.1 Core packages

```bash
pip install \
    opencv-python-headless==4.9.0.80 \
    numpy>=1.24,<2.0 \
    pyttsx3==2.98 \
    requests>=2.31 \
    pyaudio==0.2.14 \
    vosk==0.3.45
```

> **Note on `opencv-python-headless`:** Use the `-headless` variant on RPi — it doesn't need GTK/Qt display libraries. If you need `cv2.imshow()` for debugging, install `opencv-python` instead.

> **Note on `pyaudio`:** Requires `portaudio19-dev` (installed in step 4). If `pip install pyaudio` fails, try:
> ```bash
> sudo apt install python3-pyaudio
> # Then inside venv:
> pip install pyaudio --no-build-isolation
> ```

### 6.2 YOLO / Ultralytics

```bash
pip install ultralytics==8.2.0
```

On first run, the script will auto-download `yolov8n.pt` (~6 MB). To pre-download:

```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

> **RPi compatibility:** Ultralytics v8.2+ supports ARM64. If you see torch errors, install PyTorch for ARM:
> ```bash
> pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
> ```
> Using the CPU wheel keeps RAM usage low.

### 6.3 Verify installation

```bash
python3 -c "
import cv2; print('OpenCV', cv2.__version__)
import numpy; print('NumPy', numpy.__version__)
import pyttsx3; print('pyttsx3 OK')
import requests; print('requests OK')
import pyaudio; print('PyAudio OK')
from vosk import Model; print('Vosk OK')
from ultralytics import YOLO; print('Ultralytics OK')
print('All packages verified ✓')
"
```

---

## 7. Camera Configuration

### 7.1 USB Webcam (Recommended)

Your USB webcam ("Hangzhou Riyue Electronic Co., Ltd. USB2.0 Camera") should work out of the box:

```bash
# Verify detection
lsusb
# Should show: Bus 001 Device 003: ... USB2.0 Camera

# Check video device
ls /dev/video*
v4l2-ctl --list-devices

# Test capture
v4l2-ctl -d /dev/video0 --all
# Look for supported formats and resolutions
```

**Finding the correct camera index:**

The smart_goggles.py script automatically tries indices 0–3 with V4L2 and ANY backends. To manually find the right one:

```bash
python3 -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f'Index {i}: WORKS ({frame.shape})')
        else:
            print(f'Index {i}: opens but no frames')
        cap.release()
    else:
        print(f'Index {i}: cannot open')
"
```

**Common fix — camera permissions:**

```bash
# Add your user to the video group
sudo usermod -aG video $USER
# Log out and back in, or:
newgrp video
```

### 7.2 Raspberry Pi Camera Module (v2/v3)

```bash
# Enable legacy camera stack (needed for OpenCV access)
sudo raspi-config
# → Interface Options → Legacy Camera → Enable (if available)

# OR use libcamera stack with v4l2 adapter:
sudo modprobe bcm2835-v4l2
echo "bcm2835-v4l2" | sudo tee -a /etc/modules

# Verify
libcamera-hello --list-cameras
ls /dev/video*
```

> **Bookworm note:** On Raspberry Pi OS Bookworm, the legacy camera stack is removed. Use the `libcamera` stack with the v4l2 compat layer. The script's auto-detection handles this.

### 7.3 ESP32-CAM (Network Camera)

If using an ESP32-CAM over Wi-Fi, you'll need to modify the camera code to use an HTTP stream instead of a local device. Replace `SharedCamera._open_camera()` logic with:

```python
# In CAMERA_INDICES_TO_TRY, you can add the stream URL:
# cap = cv2.VideoCapture("http://192.168.1.100:81/stream")
```

This is not included by default because the latency is generally too high for real-time obstacle detection. Use a directly connected camera for safety-critical operation.

---

## 8. Audio & Microphone Configuration

### 8.1 Speaker output

```bash
# List playback devices
aplay -l

# Set default output (0 = auto, 1 = headphone jack, 2 = HDMI)
# For 3.5mm headphone jack:
amixer cset numid=3 1

# Test
espeak "Testing one two three"
speaker-test -t wav -c 2
```

### 8.2 Microphone

```bash
# List recording devices
arecord -l

# Test recording (5 seconds)
arecord -d 5 -f S16_LE -r 16000 -c 1 test.wav
aplay test.wav

# Adjust microphone volume
alsamixer
# Use F6 to select the USB audio device, adjust capture volume
```

### 8.3 PulseAudio (if needed)

```bash
# Start PulseAudio
pulseaudio --start

# List sources and sinks
pactl list sources short
pactl list sinks short

# Set default source (microphone)
pactl set-default-source <source-name>
```

### 8.4 ALSA configuration for USB audio

If the Pi defaults to HDMI audio, create `~/.asoundrc`:

```bash
cat > ~/.asoundrc << 'EOF'
pcm.!default {
    type hw
    card 1
}
ctl.!default {
    type hw
    card 1
}
EOF
```

Change `card 1` to match your USB audio device number from `aplay -l`.

---

## 9. Vosk Speech Model Setup

```bash
# Download the small English model (~40 MB)
cd ~
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
rm vosk-model-small-en-us-0.15.zip

# Verify
ls ~/vosk-model-small-en-us-0.15/
# Should contain: conf/ graph/ am/ ivector/ ...
```

The script expects the model at `~/vosk-model-small-en-us-0.15`. If you place it elsewhere, update `VOSK_MODEL_PATH` in the config section at the top of `smart_goggles.py`.

---

## 10. API Key Configuration

### 10.1 OpenRouter API Key

1. Go to [https://openrouter.ai](https://openrouter.ai)
2. Sign up / log in
3. Go to **Keys** → **Create Key**
4. Copy the key (starts with `sk-or-...`)

### 10.2 Set the environment variable

```bash
# Add to .bashrc for persistence
echo 'export OPENROUTER_API_KEY="sk-or-your-key-here"' >> ~/.bashrc
source ~/.bashrc

# Verify
echo $OPENROUTER_API_KEY
```

### 10.3 Test the API

```bash
curl -s https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" | python3 -m json.tool | head -20
```

> **Cost note:** The default model `google/gemini-flash-1.5` is very affordable. At ~30 KB per image and 200 max tokens, each query costs well under $0.01. Budget $5–10 for extensive testing.

---

## 11. Running the Application

### 11.1 Quick start

```bash
source ~/goggles-env/bin/activate
cd ~/HarithaProject   # or wherever smart_goggles.py is located
python3 smart_goggles.py
```

### 11.2 What to expect on startup

```
Smart goggles activated. Initialising camera.
Trying camera index 0 with backend V4L2 …
✓ Camera opened: index=0 backend=V4L2 resolution=640x480
Camera ready.
Loading YOLO model 'yolov8n.pt' …
YOLO model loaded and warmed up
Obstacle detection active.
Vosk speech model loaded
Voice assistant ready. Say hello to ask a question.
```

### 11.3 Using the system

- **Obstacle alerts** happen automatically — the system announces detected objects with their direction (clock position) and estimated distance
- **To ask a question:** Say **"hello"**, wait for "Ready for your question", then ask your question naturally (e.g., "What's in front of me?", "Can you read that sign?", "Is it safe to cross?")
- **To stop:** Press `Ctrl+C`

### 11.4 Logging

- Console output shows INFO-level logs
- Full logs are written to `smart_goggles.log` in the working directory
- For more detail: change `LOG_LEVEL = "DEBUG"` at the top of the script

---

## 12. Auto-Start on Boot (Production)

### 12.1 Create a systemd service

```bash
sudo nano /etc/systemd/system/smartgoggles.service
```

Paste:

```ini
[Unit]
Description=Smart Goggles AI Navigation System
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/home/pi/HarithaProject
Environment=OPENROUTER_API_KEY=sk-or-your-key-here
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/pi/goggles-env/bin/python3 /home/pi/HarithaProject/smart_goggles.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
MemoryMax=3G
CPUQuota=90%

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=/home/pi/HarithaProject

[Install]
WantedBy=multi-user.target
```

> **Important:** Replace `pi` with your actual username and update paths as needed.

### 12.2 Enable and start

```bash
sudo systemctl daemon-reload
sudo systemctl enable smartgoggles.service
sudo systemctl start smartgoggles.service
```

### 12.3 Manage the service

```bash
# Check status
sudo systemctl status smartgoggles.service

# View live logs
journalctl -u smartgoggles.service -f

# Stop
sudo systemctl stop smartgoggles.service

# Disable auto-start
sudo systemctl disable smartgoggles.service

# Restart after code changes
sudo systemctl restart smartgoggles.service
```

### 12.4 Ensure audio works in systemd

systemd services run without a desktop session, so PulseAudio may not be available. Add to the service file:

```ini
[Service]
Environment=PULSE_SERVER=unix:/run/user/1000/pulse/native
Environment=XDG_RUNTIME_DIR=/run/user/1000
```

Or switch to ALSA-only TTS by using espeak directly (the script already falls back to this).

---

## 13. Configuration Tuning

All configuration is at the top of `smart_goggles.py`. Key parameters to tune:

### Camera Performance

| Parameter | Default | Notes |
|-----------|---------|-------|
| `CAMERA_WIDTH` | 640 | Lower = faster but less detail |
| `CAMERA_HEIGHT` | 480 | Match your camera's native ratio |
| `CAMERA_FPS` | 15 | USB cams on RPi often max at 15-30 |

### YOLO Performance vs Accuracy

| Parameter | Default | Faster → | More Accurate → |
|-----------|---------|----------|-----------------|
| `YOLO_PROCESS_WIDTH` | 320 | 160 | 480 |
| `YOLO_PROCESS_HEIGHT` | 240 | 120 | 360 |
| `YOLO_CONFIDENCE` | 0.40 | 0.30 | 0.55 |
| `YOLO_FRAME_INTERVAL` | 0.25s | 0.5s | 0.15s |
| `YOLO_ALERT_COOLDOWN` | 5.0s | 3.0s | 8.0s |

### LLM Tuning

| Parameter | Default | Notes |
|-----------|---------|-------|
| `LLM_MODEL` | `google/gemini-flash-1.5` | Fastest. Alt: `openai/gpt-4o-mini` |
| `LLM_IMAGE_WIDTH` | 384 | Smaller = faster upload |
| `LLM_JPEG_QUALITY` | 30 | Lower = smaller file, less detail |
| `LLM_TIMEOUT_S` | 20 | Increase on slow networks |

### Speech

| Parameter | Default | Notes |
|-----------|---------|-------|
| `TTS_RATE` | 160 | WPM. Range: 100 (slow) – 200 (fast) |
| `LISTEN_TIMEOUT_S` | 7.0 | Max question recording length |
| `SILENCE_TIMEOUT_S` | 2.0 | Silence before stopping recording |

---

## 14. Troubleshooting

### 14.1 Camera Problems

#### "No working camera found"

```bash
# 1. Check if the camera is detected
lsusb
ls /dev/video*
v4l2-ctl --list-devices

# 2. Check permissions
groups $USER   # should include 'video'
sudo usermod -aG video $USER && newgrp video

# 3. Check if another process is using the camera
sudo fuser /dev/video0

# 4. Try manual capture
python3 -c "
import cv2
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
print('Opened:', cap.isOpened())
ret, f = cap.read()
print('Read:', ret, f.shape if ret else 'failed')
cap.release()
"

# 5. RPi Camera Module specific
vcgencmd get_camera   # should show detected=1
sudo modprobe bcm2835-v4l2
```

#### "VIDEOIO V4L2 can't open camera by index"

This is the error you described. Solutions:

1. **Wrong index:** The script tries 0–3 automatically. If your camera is at a higher index, edit `CAMERA_INDICES_TO_TRY`
2. **Permission denied:** Run `sudo usermod -aG video $USER` and re-login
3. **Camera busy:** Run `sudo fuser /dev/video*` to find and kill blocking processes
4. **V4L2 not loaded:** Run `sudo modprobe v4l2_common`
5. **USB power:** Some cameras need more power than a single USB port provides. Use a powered USB hub

#### Camera works but frames are black/green

```bash
# Camera may need warm-up time. Increase:
# CAMERA_WARMUP_FRAMES = 15  (in smart_goggles.py)

# Or check exposure settings:
v4l2-ctl -d /dev/video0 -C exposure_auto
v4l2-ctl -d /dev/video0 -c exposure_auto=3  # auto exposure
```

### 14.2 Audio Problems

#### No speech output

```bash
# 1. Test espeak directly
espeak "Test"

# 2. Check audio output device
aplay -l
amixer cset numid=3 1   # force 3.5mm jack

# 3. Check volume
amixer get Master
amixer set Master 100%

# 4. Test with a WAV file
speaker-test -t wav

# 5. Check PulseAudio
pulseaudio --check
pulseaudio --start
pactl list sinks short
```

#### Microphone not working

```bash
# 1. List recording devices
arecord -l

# 2. Test recording
arecord -d 3 -f S16_LE -r 16000 -c 1 /tmp/test.wav
aplay /tmp/test.wav

# 3. Adjust mic volume
alsamixer   # press F6 to select USB device, F4 for capture

# 4. Check PyAudio sees the mic
python3 -c "
import pyaudio
pa = pyaudio.PyAudio()
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f'Input {i}: {info[\"name\"]}')
pa.terminate()
"
```

### 14.3 YOLO / Performance Problems

#### "Failed to load YOLO model"

```bash
# 1. Check disk space
df -h

# 2. Re-download model
python3 -c "from ultralytics import YOLO; m = YOLO('yolov8n.pt'); print('OK')"

# 3. Check PyTorch
python3 -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

#### YOLO is too slow / high CPU

```bash
# 1. Monitor resources
htop

# 2. Reduce inference resolution (in smart_goggles.py):
# YOLO_PROCESS_WIDTH = 160
# YOLO_PROCESS_HEIGHT = 120

# 3. Increase frame interval:
# YOLO_FRAME_INTERVAL = 0.5  (2 FPS instead of 4)

# 4. Check CPU temperature (throttling?)
vcgencmd measure_temp
# If > 80°C, add cooling (heatsink/fan)

# 5. Check CPU frequency (throttled?)
vcgencmd get_throttled
# 0x0 = no throttling
```

#### Out of memory

```bash
# 1. Check memory
free -h

# 2. Add swap space
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 3. Reduce GPU memory
echo "gpu_mem=64" | sudo tee -a /boot/firmware/config.txt
sudo reboot
```

### 14.4 Network / LLM API Problems

#### "LLM API connection error"

```bash
# 1. Check internet
ping -c 3 google.com
curl -I https://openrouter.ai

# 2. Check DNS
nslookup openrouter.ai

# 3. Check API key
echo $OPENROUTER_API_KEY

# 4. Test API directly
curl -s https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" | head -5
```

#### "LLM API timeout"

- Increase `LLM_TIMEOUT_S` (default: 20s)
- Check your internet speed: `speedtest-cli`
- Use a faster model: `google/gemini-flash-1.5` is fastest
- Reduce image size: lower `LLM_IMAGE_WIDTH` / `LLM_JPEG_QUALITY`

#### Rate limiting (429 errors)

- The script automatically retries after 3s
- Add credits to your OpenRouter account
- Monitor usage at https://openrouter.ai/activity

### 14.5 Vosk / Wake-Word Problems

#### "Vosk model not found"

```bash
# Verify model path
ls ~/vosk-model-small-en-us-0.15/

# If missing, re-download
cd ~
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

#### Wake word not detected

- Speak clearly and at moderate volume
- Ensure the microphone is working (test with `arecord`)
- Try in a quieter environment
- Check logs for speech recognition output (set `LOG_LEVEL = "DEBUG"`)
- The Vosk small model has limited accuracy — try the medium model:
  ```bash
  wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
  # Update VOSK_MODEL_PATH accordingly (larger model, better accuracy, ~1GB)
  ```

### 14.6 General Debugging

```bash
# Run with debug logging
# Edit smart_goggles.py: LOG_LEVEL = "DEBUG"
python3 smart_goggles.py 2>&1 | tee debug.log

# Check the log file
tail -100 smart_goggles.log

# Monitor system resources during operation
htop
# or
watch -n 2 'free -h; vcgencmd measure_temp'
```

---

## 15. Architecture Reference

### Threading Model

```
Main Thread
 │
 ├── AudioManager (TTS Thread)
 │    └── Priority queue: System > LLM > YOLO
 │    └── YOLO alerts are interruptible
 │
 ├── YOLODetector (Detection Thread)
 │    └── Continuous loop: grab frame → resize → infer → announce
 │    └── Never paused, even during LLM queries
 │    └── Cooldown prevents repeating same alert
 │
 ├── VisionAssistant (VQA Thread)
 │    └── Wake-word loop: listen → "hello" → record question
 │    └── Grabs one frame → encodes JPEG → calls LLM API
 │    └── Speaks answer through AudioManager
 │
 └── SharedCamera
      └── Thread-safe wrapper around cv2.VideoCapture
      └── Auto-recovery on camera disconnect
      └── Single lock protects all read() calls
```

### Data Flow

```
Camera → [SharedCamera.read()] → YOLO thread → detections → AudioManager
                                                                 ↓
Camera → [SharedCamera.grab_for_llm()] → VQA thread → LLM API → AudioManager
                                                                     ↓
                                                              espeak / pyttsx3
                                                                     ↓
                                                                  Speaker
```

### File Structure

```
~/HarithaProject/
├── smart_goggles.py          # Main application (single file)
├── smart_goggles.log         # Runtime log (auto-created)
├── SETUP_GUIDE.md            # This guide
└── yolov8n.pt                # YOLO model (auto-downloaded on first run)

~/vosk-model-small-en-us-0.15/  # Vosk speech recognition model
~/goggles-env/                   # Python virtual environment
```

---

## Quick Reference Card

| Action | Command |
|--------|---------|
| Start | `python3 smart_goggles.py` |
| Stop | `Ctrl+C` |
| Ask a question | Say "hello" → wait for beep → ask |
| View logs | `tail -f smart_goggles.log` |
| Service start | `sudo systemctl start smartgoggles` |
| Service stop | `sudo systemctl stop smartgoggles` |
| Service logs | `journalctl -u smartgoggles -f` |
| Check camera | `v4l2-ctl --list-devices` |
| Test audio | `espeak "test"` |
| Test mic | `arecord -d 3 test.wav && aplay test.wav` |

---

*Last updated: February 2026*

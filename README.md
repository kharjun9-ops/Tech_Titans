================================================================================
PROJECT TITLE
================================================================================
Multimodal AI for Real-Time Threat Detection (Visual + Audio Fusion)

================================================================================
PROJECT OVERVIEW
================================================================================
This is an intelligent surveillance system that combines computer vision (camera)
and audio analytics (microphone) to detect suspicious activities in real-time.

Unlike normal CCTV or sound alarms, this system uses MULTIMODAL FUSION - it only
triggers an alert when BOTH visual AND audio anomalies happen within a short
time window (e.g., 5-10 seconds). This drastically reduces false alarms.

================================================================================
IOT / MULTI-SENSOR FUSION (ROADMAP)
================================================================================

To extend beyond audio+video, the same fusion engine can ingest low-cost IoT sensors
to make alerts more reliable and deployment-ready:

Sensors to add:
- Door sensor (open/close)
- PIR motion sensor (presence)
- Ambient light sensor (lights on/off)
- Vibration sensor (forced entry / impact)

Example fusion rules:
- Door opened + no person detected (camera) → anomaly
- Door opened after-hours + aggressive audio → high priority
- Lights turned off + motion detected → suspicious activity

Implementation idea:
- A small “sensor events” input (MQTT/HTTP/WebSocket/serial) that produces events
	in the same format as audio/visual events, then reuses the existing fusion window.

================================================================================
HOW IT WORKS (3 STEPS)
================================================================================

STEP 1: INDEPENDENT DETECTION
-----------------------------------------
VISUAL DETECTION (Camera):
- Shoplifting gestures (concealing items)
- Loitering near restricted areas
- Running inside stores / banks
- Crowd stampede or fight detection
- Masked person in sensitive zones
- Weapon-like objects (gun, knife, etc.)

AUDIO DETECTION (Microphone):
- Screaming / distress sounds (especially at night)
- Glass breaking (burglary detection)
- Gunshot sounds
- Aggressive shouting / altercation
- Silence after alarm trigger (confirms event)

STEP 2: MULTIMODAL FUSION
-----------------------------------------
Both detection systems run simultaneously. An alert is triggered ONLY IF:
- A visual anomaly is detected AND
- An audio anomaly is detected within 5-10 seconds of each other

Example: Masked person (visual) + Glass breaking (audio) within 7 seconds = REAL THREAT

STEP 3: WHATSAPP ALERT
-----------------------------------------
When fusion triggers:
- System captures a short video clip (10 seconds before and after event)
- Sends clip via WhatsApp API to security team/user
- Clip includes timestamp and type of anomalies detected

================================================================================
FEATURES
================================================================================
✓ Real-time visual and audio monitoring
✓ Low false alarm rate (fusion-based)
✓ WhatsApp integration with video evidence
✓ Works in retail stores, banks, schools, warehouses
✓ Night mode for audio detection (screaming focus)
✓ Silence detection after alarm (confirms incident)

================================================================================
PROJECT STRUCTURE
================================================================================
/visual-detection    - Camera & image processing code
/audio-detection     - Microphone & sound analysis code
/fusion-engine       - Combines visual + audio triggers
/whatsapp-alerts     - WhatsApp API and clip sending
/models              - Trained AI models (YOLO, audio)
/datasets            - Training data
/logs                - Event logs
/output-clips        - Saved video clips of incidents

================================================================================
TECHNOLOGY STACK
================================================================================
- Python (main language)
- OpenCV (video processing)
- YOLO (object/gesture detection)
- MFCC / Librosa (audio feature extraction)
- Twilio API (WhatsApp messaging)
- PyTorch / TensorFlow (AI models)

================================================================================
INSTALLATION (TO BE ADDED)
================================================================================
[Installation steps will be added here after development]

Basic requirements:
- Python 3.8+
- Webcam / IP camera
- USB Microphone
- WhatsApp Business API access

================================================================================
CURRENT STATUS
================================================================================
Project: IN DEVELOPMENT

Completed:
- Project structure created
- README documentation

In Progress:
- Error alert sound selection (see progress.txt)
- Visual detection module
- Audio detection module
- Fusion engine
- WhatsApp integration

================================================================================
LOCAL ALERT SOUND
================================================================================

When an alert triggers, the app can play a local alarm sound on this PC.

- ENABLE_ALARM_SOUND=1  (turn on)
- ALARM_STYLE=beep      (default) or siren

================================================================================
QUICKSTART (DEMO)
================================================================================

1) Install deps:
	pip install -r requirements.txt

2) (Optional) Create a .env from .env.example and set:
	CAMERA_INDEX=0
	SEND_WHATSAPP=0

3) Run:
	python src/app.py

Hotkeys:
- h : toggle HUD
- b : toggle boxes
- r : set restricted zone (then click 2 corners)
- q : quit

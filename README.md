# Multimodal AI for Real-Time Threat Detection

An intelligent surveillance and incident-response demo that fuses **camera vision**, **microphone audio**, and optional **local alarm + WhatsApp evidence delivery** into one real-time safety pipeline.

## Problem Statement

Most security systems are built around a single signal. A camera may see a person, an audio detector may hear a sound, and a guard may receive too many separate alerts to trust any of them. That creates two failures:

1. False alarms from isolated events that are not actually dangerous.
2. Slow response when a real threat is spread across multiple signals.

This project solves that by using **multimodal fusion**. Instead of firing on one weak cue, the system waits for a visual event and an audio event to appear within a short time window before it escalates. The result is a smarter, more credible alert stream for schools, retail stores, warehouses, banks, and other monitored spaces.

## Project Description

This repository contains a real-time threat-detection prototype built in Python with OpenCV and YOLO-based vision analysis. The app continuously reads from a camera feed, tracks suspicious activity patterns, listens for audio anomalies, and stores short incident clips for evidence.

The system is designed to feel like a practical security product rather than a one-off model demo. It includes a clean on-screen HUD, configurable thresholds through environment variables, local model resolution so the repo weights are used directly, and a local alarm sound that can play on the demo machine when an alert triggers.

## What It Detects

Visual events:

- Restricted-zone intrusion
- Loitering
- Running or sudden movement
- Crowd escalation or fight-like motion
- Masked presence
- Weapon-like object confirmation
- Shoplifting-style activity patterns

Audio events:

- Screaming or distress sounds
- Aggressive shouting
- Gunshot-like impulses
- Glass-breaking-like transients
- Silence after an incident, used as a confirmatory signal

## How the Fusion Works

1. The camera and microphone run independently.
2. Each subsystem emits timestamped anomaly events with a score.
3. The fusion layer waits for matching evidence inside a configurable window.
4. When the signal combination is strong enough, the system triggers:
	- a local alarm sound,
	- a short clip capture,
	- and optionally a WhatsApp alert with media evidence.

This makes the alert behavior stricter than a normal CCTV trigger and much harder to spoof with a single noisy event.

## Key Features

- Real-time video and audio monitoring
- Fusion-based alerting to reduce false positives
- Short incident clip capture with timestamped evidence
- WhatsApp alert workflow for remote notification
- Local Windows alarm sound with selectable style
- Demo HUD with FPS, status, and anomaly labels
- Restricted-zone calibration from the live view
- Configurable thresholds through `.env`
- Local YOLO model path resolution to avoid surprise downloads

## Demo Controls

- `h` toggles the HUD
- `b` toggles detection boxes
- `r` starts restricted-zone calibration, then click 2 corners
- `q` quits the application

## Setup

1. Install dependencies:

	```bash
	pip install -r requirements.txt
	```

2. Create a `.env` file from `.env.example` and adjust the values for your machine.

3. Run the app:

	```bash
	python src/app.py
	```

## Important Environment Settings

- `ENABLE_AUDIO=1` enables the microphone pipeline.
- `REQUIRE_FUSION=1` keeps alerts strict and requires both audio and visual evidence.
- `ENABLE_ALARM_SOUND=1` turns on the local alarm sound.
- `ALARM_STYLE=beep` or `ALARM_STYLE=siren` chooses the alert tone.
- `SEND_WHATSAPP=1` enables the WhatsApp media workflow.
- `PUBLIC_CLIP_BASE_URL` should point to your public clip URL if you use ngrok.

## Tech Stack

- Python
- OpenCV
- YOLO-based vision detection
- Real-time audio analysis
- Twilio WhatsApp integration
- FFmpeg for clip transcoding

## Repository Layout

- `src/app.py` main real-time application loop
- `src/audio.py` audio anomaly detection
- `src/vision.py` camera analysis and event logic
- `src/config.py` environment-driven configuration
- `src/alert.py` alert delivery helpers
- `clips/` saved incident clips
- `IOT_SENSORS_ROADMAP.md` future sensor-fusion expansion plan

## Roadmap

The same fusion pattern can be extended to IoT sensors such as door contacts, PIR motion, light sensors, and vibration sensors. That would let the system combine camera, audio, and physical-world signals into a single decision engine for higher-confidence security automation.

## Current Status

The core demo pipeline is implemented and ready for presentation: live detection, fusion logic, evidence capture, alert sound, and configuration are all in place.

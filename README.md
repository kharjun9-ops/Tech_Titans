# Multimodal AI for Real-Time Threat Detection (Visual + Audio Fusion)

Real-time surveillance system combining computer vision and audio analytics to detect threats (shoplifting, weapons, fights, glass breaks, gunshots, screams, etc.).  
Reduces false alarms via multimodal fusion — triggers WhatsApp alert with video clip only when both visual AND audio anomalies occur within a short window.

---

## Multimodal Surveillance Demo

This is a hackathon demo that combines live webcam + microphone detection.  
It fires an alert only when a visual anomaly and an audio anomaly happen within a short window, reducing false alarms.

---

## What it does (MVP)

### Visual Detection:
- Weapon-like objects  
- Running motion  
- Crowd/fight detection  
- Loitering in restricted zone  
- Masked person in restricted zone  
- Possible concealment near shelf  

### Audio Detection:
- Scream/distress  
- Glass breaking  
- Gunshot  
- Aggressive shouting  
- Silence after alarm  

### Fusion Logic:
- Alert triggers **only when BOTH audio + visual events occur together**

### Output:
- Console alerts (default)  
- Optional WhatsApp alerts via Twilio  
- Saves last few seconds video clip in `clips/`

---

## Notes
- Some visual detections are heuristic (for demo simplicity)
- Uses **YAMNet** for audio classification  
- Falls back to loudness detection if model fails  

---

## 🚀 Quick Start

### 1. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup environment variables
```bash
copy .env.example .env
```

### 4. Run the app
```bash
python src/app.py
```

Press **q** to quit.

---

## 📲 WhatsApp Alert Setup (Optional)

1. Create a Twilio account  
2. Enable WhatsApp sandbox  
3. Add to `.env`:

```
SEND_WHATSAPP=1
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
WHATSAPP_FROM=whatsapp:+14155238886
WHATSAPP_TO=whatsapp:+91XXXXXXXXXX
```

---

## 🎥 Send Clip via WhatsApp (Optional)

### Start local server:
```bash
cd clips
python -m http.server 8000
```

### Start ngrok:
```bash
ngrok http 8000
```

### Then set in `.env`:
```
PUBLIC_CLIP_BASE_URL=https://your-ngrok-url
```

---

## 🎯 Demo Tips

- Use good lighting for webcam  
- Play audio clips near mic to simulate events  
- Define restricted zones clearly  

---

## 💡 Project Goal

Reduce false alarms in surveillance systems using **multimodal AI fusion (Vision + Audio)**.

---

## 👨‍💻 Built for Hackathon
A fast MVP demonstrating real-time threat detection with practical deployment options.
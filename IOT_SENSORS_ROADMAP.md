# IoT Sensors Roadmap (Future Extension)

This project already fuses **visual (camera)** + **audio (microphone)** events.
A strong real-world deployment story is to extend the same fusion engine to include
low-cost IoT sensors.

## Sensors to integrate
- **Door contact sensor**: open/close events
- **PIR motion sensor**: presence/motion in corridor/entry
- **Light sensor**: lights on/off (useful for night scenarios)
- **Vibration/impact sensor**: forced entry, breaking/glass impact proxy

## Example fusion rules
- Door opened + **no person detected** within N seconds → suspicious
- Door opened after-hours + **aggressive shouting** → high priority
- Lights turned off + motion detected → suspicious
- Impact detected + **weapon confirmed** → immediate alert

## Minimal event schema (compatible with existing fusion)

Each sensor becomes a producer of events like:

```json
{
  "type": "sensor",
  "label": "door_open",
  "score": 1.0,
  "time": 1714250000.0,
  "meta": {
    "device_id": "door-1",
    "location": "front_entrance"
  }
}
```

Then the fusion engine can treat `sensor` exactly like `audio`/`visual`:
- same time-window logic
- same cooldown logic
- same alert pipeline (clip + WhatsApp + local alarm)

## Transport options (choose 1)
- MQTT (most common for IoT)
- Simple HTTP POST endpoint (easy for hackathon demos)
- Serial/USB (Arduino)
- WebSocket stream

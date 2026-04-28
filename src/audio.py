import time
from typing import Dict, List
from urllib.request import urlopen

import numpy as np


class AudioDetector:
    def __init__(self, config) -> None:
        self.config = config
        self.sample_rate = config.audio_sample_rate
        self.window_samples = int(self.sample_rate * config.audio_window_sec)
        self.hop_samples = int(self.sample_rate * config.audio_hop_sec)
        self.score_threshold = config.audio_score_threshold
        self.min_hits = config.audio_min_hits
        self.hit_window_sec = config.audio_hit_window_sec
        self.loud_rms_threshold = config.audio_loud_rms_threshold
        self.silence_rms_threshold = config.silence_rms_threshold
        self.silence_confirm_sec = config.silence_confirm_sec
        self.event_cooldown_sec = config.event_cooldown_sec
        self.last_event_times: Dict[str, float] = {}
        self.hit_times: Dict[str, List[float]] = {}
        self.last_alarm_time = 0.0
        self.model_loaded = False
        self.load_error = ""
        self.class_names: List[str] = []
        self.targets = {
            "scream/distress": ["scream", "screaming", "yell"],
            "glass breaking": ["glass", "breaking", "shatter"],
            "gunshot": ["gunshot", "gunfire"],
            "aggressive shouting": ["shout", "yell", "scream"],
            "silence after alarm": ["silence"],
        }
        self._load_model()

    def _load_model(self) -> None:
        try:
            import tensorflow as tf
            import tensorflow_hub as hub

            self.tf = tf
            self.model = hub.load("https://tfhub.dev/google/yamnet/1")
            self.class_names = self._load_class_map()
            self.model_loaded = True
        except Exception as exc:
            self.load_error = str(exc)
            self.model_loaded = False

    def _load_class_map(self) -> List[str]:
        url = "https://storage.googleapis.com/audioset/yamnet/yamnet_class_map.csv"
        with urlopen(url, timeout=10) as response:
            data = response.read().decode("utf-8")
        lines = data.strip().splitlines()
        labels = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) >= 3:
                labels.append(parts[2].strip().strip('"'))
        return labels

    def notify_alarm(self, alarm_time: float) -> None:
        self.last_alarm_time = alarm_time

    def run(self, event_queue, stop_event) -> None:
        if not self.config.enable_audio:
            return
        try:
            import sounddevice as sd
        except Exception as exc:
            self.load_error = str(exc)
            return

        buffer = np.zeros((0,), dtype=np.float32)

        with sd.InputStream(
            channels=1, samplerate=self.sample_rate, dtype="float32"
        ) as stream:
            while not stop_event.is_set():
                data, _ = stream.read(self.hop_samples)
                samples = data[:, 0]
                buffer = np.concatenate([buffer, samples])

                if len(buffer) < self.window_samples:
                    continue

                window = buffer[: self.window_samples]
                buffer = buffer[self.hop_samples :]

                events = self._detect(window)
                for event in events:
                    event_queue.put(event)

    def _detect(self, audio_samples: np.ndarray) -> List[dict]:
        now = time.time()
        events: List[dict] = []

        rms = float(np.sqrt(np.mean(np.square(audio_samples))))
        if self.last_alarm_time > 0:
            if now - self.last_alarm_time <= self.silence_confirm_sec:
                if rms < self.silence_rms_threshold:
                    if self._register_hit("silence", now) and self._event_ready("silence", now):
                        events.append(self._make_event("audio", "silence after alarm", 0.6))
                        self._clear_hits("silence")

        if not self.model_loaded:
            if (
                rms > self.loud_rms_threshold
                and self._register_hit("loud", now)
                and self._event_ready("loud", now)
            ):
                events.append(self._make_event("audio", "aggressive shouting", rms))
                self._clear_hits("loud")
            return events

        audio_tensor = self.tf.convert_to_tensor(audio_samples, dtype=self.tf.float32)
        scores, _, _ = self.model(audio_tensor)
        scores = scores.numpy().mean(axis=0)

        top_indices = scores.argsort()[-8:][::-1]
        for idx in top_indices:
            label = self.class_names[idx].lower() if idx < len(self.class_names) else ""
            score = float(scores[idx])
            if score < self.score_threshold:
                continue
            for event_label, keywords in self.targets.items():
                if any(k in label for k in keywords):
                    if self._register_hit(event_label, now) and self._event_ready(
                        event_label, now
                    ):
                        events.append(self._make_event("audio", event_label, score, label))
                        self._clear_hits(event_label)
        return events

    def _register_hit(self, key: str, now: float) -> bool:
        hits = self.hit_times.get(key, [])
        hits = [t for t in hits if now - t <= self.hit_window_sec]
        hits.append(now)
        self.hit_times[key] = hits
        return len(hits) >= self.min_hits

    def _clear_hits(self, key: str) -> None:
        self.hit_times.pop(key, None)

    def _event_ready(self, key: str, now: float) -> bool:
        last_time = self.last_event_times.get(key)
        if last_time is None or (now - last_time) >= self.event_cooldown_sec:
            self.last_event_times[key] = now
            return True
        return False

    @staticmethod
    def _make_event(source: str, label: str, score: float, detail: str = "") -> dict:
        return {
            "type": source,
            "label": label,
            "score": score,
            "detail": detail,
            "time": time.time(),
        }

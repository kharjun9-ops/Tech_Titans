import math
import os
import shutil
import subprocess
import queue
import threading
import time
import wave
from array import array
from collections import deque
from typing import List, Optional, Tuple

import cv2

from alert import send_alert
from audio import AudioDetector
from config import Config
from vision import VisionDetector


class FrameBuffer:
    def __init__(self, config: Config, fallback_fps: Optional[float] = None) -> None:
        self.clip_seconds = float(config.clip_seconds)
        self.frames = deque()
        self.timestamps = deque()
        self.lock = threading.Lock()
        self.size = (config.frame_width, config.frame_height)
        self.fallback_fps = (
            fallback_fps if fallback_fps and fallback_fps >= 1 else float(config.fps)
        )
        self.clips_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "clips")
        )
        os.makedirs(self.clips_dir, exist_ok=True)

    def add(self, frame, timestamp: Optional[float] = None) -> None:
        now = timestamp if timestamp is not None else time.time()
        with self.lock:
            self.frames.append(frame.copy())
            self.timestamps.append(now)
            cutoff = now - self.clip_seconds
            while self.timestamps and self.timestamps[0] < cutoff:
                self.timestamps.popleft()
                self.frames.popleft()

    def save_clip(self) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"clip_{timestamp}.mp4"
        path = os.path.join(self.clips_dir, filename)
        temp_path = os.path.join(self.clips_dir, f"clip_{timestamp}_raw.mp4")
        with self.lock:
            frames = list(self.frames)
            timestamps = list(self.timestamps)
        if not frames:
            return ""
        fps = self._estimate_fps(timestamps)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_path, fourcc, fps, self.size)
        if not writer.isOpened():
            print("MP4 writer failed to open; sending text-only alert.")
            return ""
        for frame in frames:
            writer.write(frame)
        writer.release()
        if self._transcode_h264(temp_path, path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
            return path
        try:
            os.remove(temp_path)
        except OSError:
            pass
        return ""

    def _estimate_fps(self, timestamps: List[float]) -> float:
        if len(timestamps) < 2:
            return self.fallback_fps
        duration = timestamps[-1] - timestamps[0]
        if duration <= 0:
            return self.fallback_fps
        fps = len(timestamps) / duration
        return max(1.0, min(fps, 60.0))

    def _transcode_h264(self, source_path: str, target_path: str) -> bool:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            print("ffmpeg not found; install it to send MP4 clips on WhatsApp.")
            return False
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            source_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            target_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print("ffmpeg failed to transcode clip; sending text-only alert.")
            return False
        return True


class ZoneCalibrator:
    def __init__(self) -> None:
        self.active = False
        self.points: List[Tuple[int, int]] = []
        self.last_box: Optional[Tuple[int, int, int, int]] = None

    def start(self) -> None:
        self.active = True
        self.points.clear()

    def add_point(self, x: int, y: int) -> Optional[Tuple[int, int, int, int]]:
        if not self.active:
            return None
        self.points.append((x, y))
        if len(self.points) < 2:
            return None
        (x1, y1), (x2, y2) = self.points[:2]
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        self.active = False
        self.points.clear()
        self.last_box = (x1, y1, x2, y2)
        return self.last_box


def _draw_box(frame, box: Tuple[int, int, int, int], color: Tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def _draw_hud(
    frame,
    lines: List[str],
    origin: Tuple[int, int] = (10, 10),
    font_scale: float = 0.55,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.45,
) -> None:
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    pad_x = 10
    pad_y = 8
    line_gap = 6

    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    max_w = max(w for (w, _) in sizes)
    line_h = max(h for (_, h) in sizes)
    box_w = max_w + pad_x * 2
    box_h = pad_y * 2 + len(lines) * line_h + (len(lines) - 1) * line_gap

    x0, y0 = origin
    x1 = min(frame.shape[1] - 1, x0 + box_w)
    y1 = min(frame.shape[0] - 1, y0 + box_h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    y = y0 + pad_y + line_h
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x0 + pad_x, y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
        y += line_h + line_gap


def _draw_banner(
    frame,
    text: str,
    bg_color: Tuple[int, int, int] = (0, 0, 255),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    alpha: float = 0.35,
) -> None:
    if not text:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    pad_x = 14
    pad_y = 10

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x0 = max(0, (frame.shape[1] - (tw + pad_x * 2)) // 2)
    y0 = 10
    x1 = min(frame.shape[1] - 1, x0 + tw + pad_x * 2)
    y1 = min(frame.shape[0] - 1, y0 + th + pad_y * 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    cv2.putText(
        frame,
        text,
        (x0 + pad_x, y0 + pad_y + th),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def _ensure_siren_wav(path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)

    sample_rate = 44100
    duration_s = 1.2
    fade_s = 0.02
    freq_low = 700.0
    freq_high = 1400.0
    sweep_s = 0.25
    amplitude = 0.45

    total_samples = int(sample_rate * duration_s)
    pcm = array("h")
    phase = 0.0
    two_pi = 2.0 * math.pi

    for i in range(total_samples):
        t = i / sample_rate

        sweep_pos = (t % sweep_s) / sweep_s
        tri = 2.0 * sweep_pos if sweep_pos < 0.5 else 2.0 * (1.0 - sweep_pos)
        freq = freq_low + (freq_high - freq_low) * tri
        phase += two_pi * freq / sample_rate

        fade = 1.0
        if t < fade_s:
            fade = t / fade_s
        elif duration_s - t < fade_s:
            fade = (duration_s - t) / fade_s
        fade = max(0.0, min(1.0, fade))

        value = int(amplitude * fade * math.sin(phase) * 32767)
        pcm.append(max(-32768, min(32767, value)))

    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _play_alarm(config: Config) -> None:
    if not config.enable_alarm_sound:
        return

    style = (getattr(config, "alarm_style", "beep") or "beep").strip().lower()
    count = max(1, int(config.alarm_beep_count))
    hz = max(200, int(config.alarm_beep_hz))
    ms = max(50, int(config.alarm_beep_ms))
    try:
        import winsound

        if style == "siren":
            siren_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "clips", "alarm_siren.wav")
            )
            _ensure_siren_wav(siren_path)
            winsound.PlaySound(None, winsound.SND_PURGE)
            winsound.PlaySound(siren_path, winsound.SND_FILENAME)
            return

        for _ in range(count):
            winsound.Beep(hz, ms)
            time.sleep(0.05)
    except Exception:
        # Fallback to console bell if winsound is unavailable.
        print("\a", end="", flush=True)


def _trigger_alarm(config: Config) -> None:
    threading.Thread(target=_play_alarm, args=(config,), daemon=True).start()


def _event_recent(event: dict, now: float, window_sec: float) -> bool:
    if not event:
        return False
    return now - event["time"] <= window_sec


def _normalize_box(
    box: Tuple[int, int, int, int], frame_size: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    width, height = frame_size
    x1, y1, x2, y2 = box
    return (
        x1 / float(width),
        y1 / float(height),
        x2 / float(width),
        y2 / float(height),
    )


def main() -> None:
    config = Config.load()

    event_queue: "queue.Queue[dict]" = queue.Queue()
    stop_event = threading.Event()

    vision = VisionDetector(config)
    audio = AudioDetector(config)

    audio_thread = threading.Thread(
        target=audio.run, args=(event_queue, stop_event), daemon=True
    )
    audio_thread.start()

    cap = cv2.VideoCapture(config.camera_index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
    if not cap.isOpened():
        print(
            f"ERROR: Could not open camera index {config.camera_index}. "
            "Try changing CAMERA_INDEX in .env (e.g., 0 or 1)."
        )
        return

    camera_fps = cap.get(cv2.CAP_PROP_FPS)
    if not camera_fps or camera_fps < 1:
        camera_fps = float(config.fps)

    window_name = "Multimodal Surveillance Demo"
    cv2.namedWindow(window_name)

    calibrator = ZoneCalibrator()
    mouse_state = {
        "calibrator": calibrator,
        "config": config,
        "frame_size": (config.frame_width, config.frame_height),
    }

    def _on_mouse(event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        box = param["calibrator"].add_point(x, y)
        if box:
            normalized = _normalize_box(box, param["frame_size"])
            param["config"].restricted_zone = normalized
            print(
                "Restricted zone set: RESTRICTED_ZONE="
                f"{normalized[0]:.3f},{normalized[1]:.3f},"
                f"{normalized[2]:.3f},{normalized[3]:.3f}"
            )

    cv2.setMouseCallback(window_name, _on_mouse, mouse_state)

    buffer = FrameBuffer(config, fallback_fps=camera_fps)

    last_audio_event = None
    last_visual_event = None
    last_alert_time = 0.0
    frame_index = 0
    last_detections: List[dict] = []
    last_motion_score = 0.0
    last_person_count = 0
    show_hud = True
    show_boxes = True
    read_failures = 0
    fps_ema = 0.0
    last_perf_t = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                read_failures += 1
                if read_failures == 1:
                    print("WARNING: Camera frame read failed; retrying...")
                if read_failures >= 30:
                    print("ERROR: Camera read failed repeatedly; exiting.")
                    break
                time.sleep(0.02)
                continue
            read_failures = 0

            now_perf = time.perf_counter()
            dt = max(1e-6, now_perf - last_perf_t)
            inst_fps = 1.0 / dt
            fps_ema = inst_fps if fps_ema <= 0 else (0.9 * fps_ema + 0.1 * inst_fps)
            last_perf_t = now_perf

            frame = cv2.resize(frame, (config.frame_width, config.frame_height))

            # Keep a clean copy for clip recording (no overlays).
            raw_frame = frame.copy()

            # Use a separate frame for detection + drawing overlays.
            display_frame = frame.copy()

            frame_index += 1
            do_detect = (frame_index % max(1, config.detect_every_n)) == 0
            if do_detect:
                events, detections, motion_score, person_count = vision.detect(display_frame)
                last_detections = detections
                last_motion_score = motion_score
                last_person_count = person_count
                for event in events:
                    event_queue.put(event)
            else:
                detections = last_detections
                motion_score = last_motion_score
                person_count = last_person_count

            while True:
                try:
                    event = event_queue.get_nowait()
                except queue.Empty:
                    break
                if event["type"] == "audio":
                    last_audio_event = event
                else:
                    last_visual_event = event

            now = time.time()
            audio_ok = _event_recent(last_audio_event, now, config.fusion_window_sec)
            visual_ok = _event_recent(last_visual_event, now, config.fusion_window_sec)
            should_alert = (
                audio_ok and visual_ok
                if config.require_fusion
                else (audio_ok or visual_ok)
            )
            if should_alert and (now - last_alert_time) >= config.alert_cooldown_sec:
                if audio_ok and visual_ok:
                    reason = (
                        f"Visual: {last_visual_event['label']} | "
                        f"Audio: {last_audio_event['label']}"
                    )
                elif visual_ok:
                    reason = f"Visual: {last_visual_event['label']}"
                else:
                    reason = f"Audio: {last_audio_event['label']}"
                clip_path = buffer.save_clip()
                send_alert(reason, clip_path, config)
                last_alert_time = now
                audio.notify_alarm(now)
                _trigger_alarm(config)
                last_audio_event = None
                last_visual_event = None

            if show_boxes:
                for det in detections:
                    # Keep the live view clean: always show weapons; people boxes are optional.
                    if not det.get("is_weapon") and det.get("label") not in {"person"}:
                        continue
                    color = (0, 0, 255) if det.get("is_weapon") else (0, 255, 0)
                    _draw_box(display_frame, det["bbox"], color)
                    label = f"{det['label']} {det['conf']:.2f}"
                    cv2.putText(
                        display_frame,
                        label,
                        (det["bbox"][0], max(12, det["bbox"][1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

            restricted_box = vision.rel_box_to_abs(config.restricted_zone, display_frame.shape)
            _draw_box(display_frame, restricted_box, (0, 0, 255))

            if show_hud:
                hud_lines = [
                    f"FPS: {fps_ema:4.1f} | People: {person_count} | Motion: {motion_score:.2f}",
                    f"Detect stride: every {max(1, int(config.detect_every_n))} frame(s)",
                    f"Fusion required: {'ON' if config.require_fusion else 'OFF'} | Alarm: {'ON' if config.enable_alarm_sound else 'OFF'}",
                ]
                if last_visual_event:
                    hud_lines.append(
                        f"VISION: {last_visual_event['label']} ({last_visual_event['score']:.2f})"
                    )
                if last_audio_event:
                    hud_lines.append(
                        f"AUDIO: {last_audio_event['label']} ({last_audio_event['score']:.2f})"
                    )
                if getattr(vision, "status_labels", None):
                    hud_lines.extend(vision.status_labels)

                if calibrator.active:
                    hud_lines.append("Calibrate: click 2 corners")
                else:
                    hud_lines.append("Hotkeys: r=set zone | b=boxes | h=HUD | q=quit")

                _draw_hud(display_frame, hud_lines)

            if (time.time() - last_alert_time) <= 2.0 and last_alert_time > 0:
                _draw_banner(display_frame, "ALERT TRIGGERED")

            frame_ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if frame_ts_ms and frame_ts_ms > 0:
                buffer.add(raw_frame, timestamp=frame_ts_ms / 1000.0)
            else:
                buffer.add(raw_frame)

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                calibrator.start()
                print("Calibration: click two corners for the restricted zone.")
            if key == ord("h"):
                show_hud = not show_hud
            if key == ord("b"):
                show_boxes = not show_boxes
    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

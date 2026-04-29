import os
from dataclasses import dataclass
from typing import Tuple

from dotenv import load_dotenv


def _resolve_repo_model_path(value: str) -> str:
    """Resolve model names like 'yolov8n.pt' to a local file in the repo root if present.

    This avoids surprise downloads during the demo when the working directory isn't the
    repo root or when Ultralytics treats the string as a remote asset name.
    """
    value = (value or "").strip()
    if not value:
        return value
    if os.path.isabs(value):
        return value

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidate = os.path.join(repo_root, value)
    if os.path.exists(candidate):
        return candidate
    return value


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _env_box(name: str, default: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        return default
    try:
        return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError:
        return default


@dataclass
class Config:
    camera_index: int
    frame_width: int
    frame_height: int
    fps: int
    detect_every_n: int
    yolo_model: str
    yolo_conf: float
    yolo_imgsz: int
    weapon_min_conf: float
    weapon_min_sec: float
    restricted_zone: Tuple[float, float, float, float]
    shelf_zone: Tuple[float, float, float, float]
    loiter_sec: float
    masked_sec: float
    shoplift_sec: float
    crowd_count: int
    motion_threshold: float
    enable_motion_stabilization: bool
    stabilization_scale: float
    stabilization_min_response: float
    stabilization_max_shift_px: float
    fight_motion_threshold: float
    fight_min_people: int
    fight_min_sec: float
    fight_close_px: float
    fight_speed_px_s: float
    run_speed_px_s: float
    walk_speed_px_s: float
    run_min_sec: float
    loiter_max_speed_px_s: float
    shoplift_max_speed_px_s: float
    face_seen_confirm_sec: float
    face_min_size: int
    face_min_neighbors: int
    enable_pose: bool
    pose_model: str
    pose_conf: float
    pose_every_n: int
    enable_backpack: bool
    backpack_overlap_iou: float
    item_overlap_iou: float
    backpack_hold_sec: float
    fusion_window_sec: float
    alert_cooldown_sec: float
    clip_seconds: int
    event_cooldown_sec: float
    require_fusion: bool
    enable_audio: bool
    enable_alarm_sound: bool
    alarm_style: str
    alarm_beep_hz: int
    alarm_beep_ms: int
    alarm_beep_count: int
    audio_sample_rate: int
    audio_window_sec: float
    audio_hop_sec: float
    audio_score_threshold: float
    audio_min_hits: int
    audio_hit_window_sec: float
    audio_loud_rms_threshold: float
    silence_rms_threshold: float
    silence_confirm_sec: float
    send_whatsapp: bool
    twilio_account_sid: str
    twilio_auth_token: str
    whatsapp_from: str
    whatsapp_to: str
    public_clip_base_url: str

    @staticmethod
    def load() -> "Config":
        load_dotenv()
        return Config(
            camera_index=_env_int("CAMERA_INDEX", 0),
            frame_width=_env_int("FRAME_WIDTH", 960),
            frame_height=_env_int("FRAME_HEIGHT", 540),
            fps=_env_int("FPS", 20),
            detect_every_n=_env_int("DETECT_EVERY_N", 1),
            yolo_model=_resolve_repo_model_path(_env_str("YOLO_MODEL", "yolov8n.pt")),
            yolo_conf=_env_float("YOLO_CONF", 0.35),
            yolo_imgsz=_env_int("YOLO_IMGSZ", 640),
            weapon_min_conf=_env_float("WEAPON_MIN_CONF", 0.55),
            weapon_min_sec=_env_float("WEAPON_MIN_SEC", 0.25),
            restricted_zone=_env_box("RESTRICTED_ZONE", (0.65, 0.15, 0.95, 0.85)),
            shelf_zone=_env_box("SHELF_ZONE", (0.05, 0.2, 0.35, 0.85)),
            loiter_sec=_env_float("LOITER_SEC", 6.0),
            masked_sec=_env_float("MASKED_SEC", 3.0),
            shoplift_sec=_env_float("SHOPLIFT_SEC", 4.0),
            crowd_count=_env_int("CROWD_COUNT", 4),
            motion_threshold=_env_float("MOTION_THRESHOLD", 0.08),
            enable_motion_stabilization=_env_bool("ENABLE_MOTION_STABILIZATION", True),
            stabilization_scale=_env_float("STABILIZATION_SCALE", 0.25),
            stabilization_min_response=_env_float("STABILIZATION_MIN_RESPONSE", 0.35),
            stabilization_max_shift_px=_env_float("STABILIZATION_MAX_SHIFT_PX", 60.0),
            fight_motion_threshold=_env_float("FIGHT_MOTION_THRESHOLD", 0.14),
            fight_min_people=_env_int("FIGHT_MIN_PEOPLE", 2),
            fight_min_sec=_env_float("FIGHT_MIN_SEC", 0.6),
            fight_close_px=_env_float("FIGHT_CLOSE_PX", 120.0),
            fight_speed_px_s=_env_float("FIGHT_SPEED_PX_S", 90.0),
            run_speed_px_s=_env_float("RUN_SPEED_PX_S", 160.0),
            walk_speed_px_s=_env_float("WALK_SPEED_PX_S", 60.0),
            run_min_sec=_env_float("RUN_MIN_SEC", 0.6),
            loiter_max_speed_px_s=_env_float("LOITER_MAX_SPEED_PX_S", 35.0),
            shoplift_max_speed_px_s=_env_float("SHOPLIFT_MAX_SPEED_PX_S", 50.0),
            face_seen_confirm_sec=_env_float("FACE_SEEN_CONFIRM_SEC", 0.6),
            face_min_size=_env_int("FACE_MIN_SIZE", 28),
            face_min_neighbors=_env_int("FACE_MIN_NEIGHBORS", 4),
            enable_pose=_env_bool("ENABLE_POSE", False),
            pose_model=_resolve_repo_model_path(_env_str("POSE_MODEL", "yolov8n-pose.pt")),
            pose_conf=_env_float("POSE_CONF", 0.25),
            pose_every_n=_env_int("POSE_EVERY_N", 3),
            enable_backpack=_env_bool("ENABLE_BACKPACK", False),
            backpack_overlap_iou=_env_float("BACKPACK_OVERLAP_IOU", 0.2),
            item_overlap_iou=_env_float("ITEM_OVERLAP_IOU", 0.1),
            backpack_hold_sec=_env_float("BACKPACK_HOLD_SEC", 0.6),
            fusion_window_sec=_env_float("FUSION_WINDOW_SEC", 8.0),
            alert_cooldown_sec=_env_float("ALERT_COOLDOWN_SEC", 15.0),
            clip_seconds=_env_int("CLIP_SECONDS", 6),
            event_cooldown_sec=_env_float("EVENT_COOLDOWN_SEC", 2.0),
            require_fusion=_env_bool("REQUIRE_FUSION", False),
            enable_audio=_env_bool("ENABLE_AUDIO", True),
            enable_alarm_sound=_env_bool("ENABLE_ALARM_SOUND", True),
            alarm_style=_env_str("ALARM_STYLE", "beep").strip().lower(),
            alarm_beep_hz=_env_int("ALARM_BEEP_HZ", 1200),
            alarm_beep_ms=_env_int("ALARM_BEEP_MS", 200),
            alarm_beep_count=_env_int("ALARM_BEEP_COUNT", 3),
            audio_sample_rate=_env_int("AUDIO_SAMPLE_RATE", 16000),
            audio_window_sec=_env_float("AUDIO_WINDOW_SEC", 0.96),
            audio_hop_sec=_env_float("AUDIO_HOP_SEC", 0.48),
            audio_score_threshold=_env_float("AUDIO_SCORE_THRESHOLD", 0.14),
            audio_min_hits=_env_int("AUDIO_MIN_HITS", 2),
            audio_hit_window_sec=_env_float("AUDIO_HIT_WINDOW_SEC", 1.2),
            audio_loud_rms_threshold=_env_float("AUDIO_LOUD_RMS_THRESHOLD", 0.18),
            silence_rms_threshold=_env_float("SILENCE_RMS_THRESHOLD", 0.01),
            silence_confirm_sec=_env_float("SILENCE_CONFIRM_SEC", 4.0),
            send_whatsapp=_env_bool("SEND_WHATSAPP", False),
            twilio_account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
            twilio_auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
            whatsapp_from=os.getenv("WHATSAPP_FROM", ""),
            whatsapp_to=os.getenv("WHATSAPP_TO", ""),
            public_clip_base_url=os.getenv("PUBLIC_CLIP_BASE_URL", ""),
        )

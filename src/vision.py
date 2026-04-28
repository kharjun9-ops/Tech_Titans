import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Track:
    track_id: int
    centroid: Tuple[int, int]
    last_seen: float
    speed_px_s: float = 0.0
    running_sec: float = 0.0
    in_zone_sec: float = 0.0
    in_shelf_sec: float = 0.0
    maskless_sec: float = 0.0
    face_seen_sec: float = 0.0
    loiter_reported: bool = False
    mask_reported: bool = False
    shoplift_reported: bool = False
    running_reported: bool = False


class VisionDetector:
    def __init__(self, config) -> None:
        self.config = config
        self.model = YOLO(config.yolo_model)
        self.conf = config.yolo_conf
        self.imgsz = config.yolo_imgsz
        self.enable_pose = config.enable_pose
        self.pose_conf = config.pose_conf
        self.pose_every_n = max(1, config.pose_every_n)
        self.pose_frame_index = 0
        self.pose_results = None
        self.pose_model = YOLO(config.pose_model) if self.enable_pose else None
        self.enable_backpack = config.enable_backpack
        self.prev_gray = None
        self.last_global_shift: Tuple[float, float] = (0.0, 0.0)
        self.last_global_shift_ok: bool = False
        self.last_event_times: Dict[str, float] = {}
        self.frame_index = 0
        self.tracks: List[Track] = []
        self.next_track_id = 1
        self.track_ttl_sec = 2.5
        self.track_match_px = 60
        self.last_frame_time: float | None = None
        self.fight_sec = 0.0
        self.fight_reported = False
        self.backpack_item_sec = 0.0
        self.weapon_seen_sec = 0.0
        self.status_labels: List[str] = []
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.weapon_classes = {"knife", "scissors", "baseball bat"}
        self.backpack_labels = {"backpack", "handbag", "suitcase"}
        self.item_labels = {"bottle", "book", "cell phone", "laptop"}
        self.skeleton_pairs = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (0, 5),
            (0, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (5, 6),
            (5, 11),
            (6, 12),
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
        ]

    def rel_box_to_abs(
        self, rel_box: Tuple[float, float, float, float], frame_shape
    ) -> Tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        x1 = int(rel_box[0] * width)
        y1 = int(rel_box[1] * height)
        x2 = int(rel_box[2] * width)
        y2 = int(rel_box[3] * height)
        return x1, y1, x2, y2

    def detect(self, frame: np.ndarray) -> Tuple[List[dict], List[dict], float, int]:
        self.frame_index += 1
        now = time.time()
        dt = 0.0
        if self.last_frame_time is not None:
            dt = max(0.0, now - self.last_frame_time)
        self.last_frame_time = now
        self.status_labels = []
        events: List[dict] = []
        detections: List[dict] = []
        person_boxes: List[Tuple[int, int, int, int]] = []
        backpack_boxes: List[dict] = []
        item_boxes: List[dict] = []
        weapon_best_label = ""
        weapon_best_conf = 0.0

        results = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)
        if results:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names.get(cls_id, str(cls_id))
                det = {
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "label": label,
                    "conf": conf,
                }
                if label in self.weapon_classes:
                    det["is_weapon"] = True
                    if conf > weapon_best_conf:
                        weapon_best_conf = conf
                        weapon_best_label = label
                detections.append(det)
                if label == "person":
                    person_boxes.append(det["bbox"])
                if label in self.backpack_labels:
                    backpack_boxes.append(det)
                if label in self.item_labels:
                    item_boxes.append(det)

        motion_score = self._motion_score(frame, person_boxes)
        person_count = len(person_boxes)

        weapon_valid = (
            bool(weapon_best_label)
            and weapon_best_conf >= float(getattr(self.config, "weapon_min_conf", 0.0))
        )
        if weapon_valid:
            self.weapon_seen_sec += dt
        else:
            self.weapon_seen_sec = 0.0

        if weapon_best_label:
            if weapon_valid and self.weapon_seen_sec >= float(
                getattr(self.config, "weapon_min_sec", 0.0)
            ):
                self.status_labels.append(
                    f"Weapon CONFIRMED: {weapon_best_label} {weapon_best_conf * 100:.2f}%"
                )
            else:
                self.status_labels.append(
                    f"Weapon? {weapon_best_label} {weapon_best_conf * 100:.2f}%"
                )

        if (
            weapon_valid
            and self.weapon_seen_sec >= float(getattr(self.config, "weapon_min_sec", 0.0))
            and self._event_ready("weapon", now)
        ):
            events.append(
                self._make_event(
                    "visual", f"weapon: {weapon_best_label}", weapon_best_conf
                )
            )

        track_events = self._update_tracks(frame, person_boxes, now, motion_score, dt)
        events.extend(track_events)

        if self.enable_backpack:
            item_conf = self._item_in_backpack_conf(person_boxes, backpack_boxes, item_boxes)
            if item_conf > 0:
                self.backpack_item_sec += dt
                self.status_labels.append(f"Item in backpack: {item_conf * 100:.2f}%")
            else:
                self.backpack_item_sec = 0.0
            if (
                self.backpack_item_sec >= self.config.backpack_hold_sec
                and self._event_ready("backpack_item", now)
            ):
                events.append(self._make_event("visual", "item in backpack", item_conf))

        if self.tracks:
            fastest = max(self.tracks, key=lambda track: track.speed_px_s)
            action_label, action_score = self._action_label(fastest.speed_px_s)
            if action_label != "Standing":
                self.status_labels.insert(0, f"{action_label}: {action_score * 100:.2f}%")

        if self.enable_pose:
            self._update_pose(frame)

        return events, detections, motion_score, person_count

    def _motion_score(
        self, frame: np.ndarray, person_boxes: List[Tuple[int, int, int, int]] | None = None
    ) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.prev_gray is None:
            self.prev_gray = gray
            self.last_global_shift = (0.0, 0.0)
            self.last_global_shift_ok = False
            return 0.0

        prev_gray = self.prev_gray
        aligned_prev = prev_gray
        self.last_global_shift = (0.0, 0.0)
        self.last_global_shift_ok = False

        if bool(getattr(self.config, "enable_motion_stabilization", True)):
            scale = float(getattr(self.config, "stabilization_scale", 0.25))
            scale = max(0.1, min(1.0, scale))
            try:
                prev_small = cv2.resize(prev_gray, (0, 0), fx=scale, fy=scale)
                curr_small = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
                shift_small, response = cv2.phaseCorrelate(
                    prev_small.astype(np.float32), curr_small.astype(np.float32)
                )
                dx = float(shift_small[0]) / scale
                dy = float(shift_small[1]) / scale

                max_shift = float(getattr(self.config, "stabilization_max_shift_px", 60.0))
                min_resp = float(getattr(self.config, "stabilization_min_response", 0.35))
                if (
                    response is not None
                    and float(response) >= min_resp
                    and abs(dx) <= max_shift
                    and abs(dy) <= max_shift
                ):
                    self.last_global_shift = (dx, dy)
                    self.last_global_shift_ok = True
                    h, w = gray.shape[:2]
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    aligned_prev = cv2.warpAffine(
                        prev_gray,
                        M,
                        (w, h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
            except Exception:
                self.last_global_shift = (0.0, 0.0)
                self.last_global_shift_ok = False

        diff = cv2.absdiff(gray, aligned_prev)
        self.prev_gray = gray

        if not person_boxes:
            return float(diff.mean() / 255.0)

        total_sum = 0.0
        total_area = 0
        h, w = diff.shape[:2]
        for (x1, y1, x2, y2) in person_boxes:
            x1 = max(0, min(w - 1, int(x1)))
            y1 = max(0, min(h - 1, int(y1)))
            x2 = max(0, min(w, int(x2)))
            y2 = max(0, min(h, int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            roi = diff[y1:y2, x1:x2]
            total_sum += float(roi.sum())
            total_area += int(roi.size)

        if total_area <= 0:
            return float(diff.mean() / 255.0)
        return float((total_sum / total_area) / 255.0)

    def _update_tracks(
        self,
        frame: np.ndarray,
        person_boxes: List[Tuple[int, int, int, int]],
        now: float,
        motion_score: float,
        frame_dt: float,
    ) -> List[dict]:
        events: List[dict] = []
        if not person_boxes:
            self._purge_tracks(now)
            return events

        dx, dy = self.last_global_shift if self.last_global_shift_ok else (0.0, 0.0)
        if dx != 0.0 or dy != 0.0:
            for track in self.tracks:
                track.centroid = (int(track.centroid[0] + dx), int(track.centroid[1] + dy))

        restricted_box = self.rel_box_to_abs(self.config.restricted_zone, frame.shape)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        active_centroids: List[Tuple[int, int]] = []
        active_speeds: List[float] = []

        assigned_track_ids = set()
        for box in person_boxes:
            centroid = self._centroid(box)
            track = self._match_track(centroid, assigned_track_ids)
            if track is None:
                track = Track(self.next_track_id, centroid, now)
                self.next_track_id += 1
                self.tracks.append(track)
            assigned_track_ids.add(track.track_id)

            track_dt = max(0.0, now - track.last_seen)
            prev_centroid = track.centroid
            track.last_seen = now
            track.centroid = centroid
            if track_dt > 0:
                track.speed_px_s = self._distance(centroid, prev_centroid) / track_dt
            else:
                track.speed_px_s = 0.0
            active_centroids.append(track.centroid)
            active_speeds.append(track.speed_px_s)

            in_restricted = self._point_in_box(centroid, restricted_box)
            in_shelf = False

            if in_restricted and track.speed_px_s <= self.config.loiter_max_speed_px_s:
                track.in_zone_sec += track_dt
            else:
                track.in_zone_sec = 0.0
                track.loiter_reported = False

            track.in_shelf_sec = 0.0
            track.shoplift_reported = False

            box_w = max(0, box[2] - box[0])
            box_h = max(0, box[3] - box[1])
            face_check_ok = min(box_w, box_h) >= self.config.face_min_size * 2
            if face_check_ok:
                face_found = self._face_in_box(gray, box)
            else:
                face_found = True

            if in_restricted and face_found:
                track.face_seen_sec = min(
                    track.face_seen_sec + track_dt, self.config.face_seen_confirm_sec
                )
            elif in_restricted and not face_found:
                track.face_seen_sec = max(0.0, track.face_seen_sec - track_dt)
            else:
                track.face_seen_sec = 0.0

            face_seen_ready = track.face_seen_sec >= self.config.face_seen_confirm_sec

            if in_restricted and not face_found and face_seen_ready:
                track.maskless_sec += track_dt
            else:
                track.maskless_sec = 0.0
                track.mask_reported = False

            if track.speed_px_s >= self.config.run_speed_px_s:
                track.running_sec += track_dt
            else:
                track.running_sec = 0.0
                track.running_reported = False

            if (
                track.running_sec >= self.config.run_min_sec
                and not track.running_reported
                and self._event_ready("running", now)
            ):
                track.running_reported = True
                events.append(self._make_event("visual", "running", track.speed_px_s))

            if track.in_zone_sec >= self.config.loiter_sec and not track.loiter_reported:
                track.loiter_reported = True
                events.append(
                    self._make_event("visual", "loitering in restricted zone", 0.7)
                )

            if track.maskless_sec >= self.config.masked_sec and not track.mask_reported:
                track.mask_reported = True
                events.append(
                    self._make_event(
                        "visual", "face not visible in restricted zone", 0.7
                    )
                )



        close_people = set()
        for i in range(len(active_centroids)):
            for j in range(i + 1, len(active_centroids)):
                if (
                    self._distance(active_centroids[i], active_centroids[j])
                    <= self.config.fight_close_px
                ):
                    close_people.add(i)
                    close_people.add(j)

        close_count = len(close_people)
        fast_count = sum(
            1 for speed in active_speeds if speed >= self.config.fight_speed_px_s
        )
        if (
            frame_dt > 0
            and close_count >= self.config.fight_min_people
            and fast_count >= 2
            and motion_score >= self.config.fight_motion_threshold
        ):
            self.fight_sec += frame_dt
        else:
            self.fight_sec = 0.0
            self.fight_reported = False

        if (
            self.fight_sec >= self.config.fight_min_sec
            and not self.fight_reported
            and self._event_ready("fight", now)
        ):
            self.fight_reported = True
            events.append(self._make_event("visual", "crowd/fight", motion_score))

        self._purge_tracks(now)
        return events

    def _match_track(self, centroid: Tuple[int, int], assigned: set) -> Track:
        best_track = None
        best_dist = self.track_match_px
        for track in self.tracks:
            if track.track_id in assigned:
                continue
            dist = self._distance(centroid, track.centroid)
            if dist < best_dist:
                best_dist = dist
                best_track = track
        return best_track

    def _purge_tracks(self, now: float) -> None:
        self.tracks = [t for t in self.tracks if now - t.last_seen <= self.track_ttl_sec]

    def _action_label(self, speed_px_s: float) -> Tuple[str, float]:
        if speed_px_s >= self.config.run_speed_px_s:
            label = "Running"
        elif speed_px_s >= self.config.walk_speed_px_s:
            label = "Walking"
        else:
            return "Standing", 0.0
        score = min(0.99, speed_px_s / max(1.0, self.config.run_speed_px_s))
        return label, score

    def _item_in_backpack_conf(
        self,
        person_boxes: List[Tuple[int, int, int, int]],
        backpack_boxes: List[dict],
        item_boxes: List[dict],
    ) -> float:
        best_conf = 0.0
        for bag in backpack_boxes:
            bag_box = bag["bbox"]
            bag_conf = float(bag["conf"])
            for person_box in person_boxes:
                if self._bbox_iou(bag_box, person_box) < self.config.backpack_overlap_iou:
                    continue
                for item in item_boxes:
                    if (
                        self._bbox_iou(bag_box, item["bbox"])
                        >= self.config.item_overlap_iou
                    ):
                        best_conf = max(best_conf, min(bag_conf, float(item["conf"])))
        return best_conf

    def _update_pose(self, frame: np.ndarray) -> None:
        if not self.pose_model:
            return
        self.pose_frame_index += 1
        if self.pose_results is None or (self.pose_frame_index % self.pose_every_n) == 0:
            self.pose_results = self.pose_model.predict(
                frame, conf=self.pose_conf, verbose=False
            )
        if not self.pose_results:
            return
        self._draw_pose(frame, self.pose_results)

    def _draw_pose(self, frame: np.ndarray, pose_results) -> None:
        if not pose_results or not getattr(pose_results[0], "keypoints", None):
            return
        kpts = pose_results[0].keypoints
        try:
            points = kpts.xy.cpu().numpy()
        except Exception:
            points = kpts.xy
        conf = None
        try:
            conf = kpts.conf.cpu().numpy()
        except Exception:
            conf = getattr(kpts, "conf", None)
        boxes = None
        if getattr(pose_results[0], "boxes", None) is not None:
            try:
                boxes = pose_results[0].boxes.xyxy.cpu().numpy()
            except Exception:
                boxes = pose_results[0].boxes.xyxy

        for i, pts in enumerate(points):
            conf_i = conf[i] if conf is not None and len(conf) > i else None
            for a, b in self.skeleton_pairs:
                if a >= len(pts) or b >= len(pts):
                    continue
                if conf_i is not None and (conf_i[a] < 0.2 or conf_i[b] < 0.2):
                    continue
                p1 = (int(pts[a][0]), int(pts[a][1]))
                p2 = (int(pts[b][0]), int(pts[b][1]))
                cv2.line(frame, p1, p2, (0, 255, 255), 2)
            for j, pt in enumerate(pts):
                if conf_i is not None and conf_i[j] < 0.2:
                    continue
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (255, 0, 255), -1)

            if boxes is None or i >= len(boxes):
                continue
            x1, y1, x2, y2 = boxes[i]
            if self.tracks:
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                nearest = min(
                    self.tracks,
                    key=lambda t: self._distance(center, t.centroid),
                )
                label, score = self._action_label(nearest.speed_px_s)
                if label != "Standing":
                    cv2.putText(
                        frame,
                        f"{label}: {score * 100:.2f}%",
                        (int(x1), max(0, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )

    def _face_in_box(self, gray: np.ndarray, box: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = box
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(gray.shape[1] - 1, x2)
        y2 = min(gray.shape[0] - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return False
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        faces = self.face_cascade.detectMultiScale(
            roi,
            scaleFactor=1.1,
            minNeighbors=self.config.face_min_neighbors,
            minSize=(self.config.face_min_size, self.config.face_min_size),
        )
        return len(faces) > 0

    @staticmethod
    def _centroid(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    @staticmethod
    def _point_in_box(point: Tuple[int, int], box: Tuple[int, int, int, int]) -> bool:
        x, y = point
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    @staticmethod
    def _distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    @staticmethod
    def _bbox_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _event_ready(self, key: str, now: float) -> bool:
        last_time = self.last_event_times.get(key)
        if last_time is None or (now - last_time) >= self.config.event_cooldown_sec:
            self.last_event_times[key] = now
            return True
        return False

    @staticmethod
    def _make_event(source: str, label: str, score: float) -> dict:
        return {
            "type": source,
            "label": label,
            "score": score,
            "time": time.time(),
        }

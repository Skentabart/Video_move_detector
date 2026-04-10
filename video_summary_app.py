import os
import threading
import time
from dataclasses import dataclass
from queue import Queue, Empty
from typing import List, Optional, Tuple, Literal

import cv2
import numpy as np
from PIL import Image, ImageTk
from tkinter import Tk, StringVar, IntVar, DoubleVar, filedialog, messagebox, Toplevel, Canvas, Menu
from tkinter import ttk


@dataclass
class ProgressUpdate:
    kind: str  # "progress" | "preview" | "done" | "stopped" | "error" | "thread_finished"
    percent: Optional[float] = None
    processed: Optional[int] = None
    total: Optional[int] = None
    eta_sec: Optional[float] = None
    preview_bgr: Optional[np.ndarray] = None
    output_path: Optional[str] = None
    error: Optional[str] = None


def _safe_video_info(cap: cv2.VideoCapture) -> Tuple[float, int]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return fps, frame_count


def _overlay_from_mask(background_bgr: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    # mask01: float32 0..1
    if mask01.size == 0:
        return background_bgr
    a = np.clip(mask01, 0.0, 1.0)[..., None].astype(np.float32) * 0.85
    base = background_bgr.astype(np.float32)
    red = np.zeros_like(base)
    red[..., 2] = 255.0
    out = base * (1.0 - a) + red * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_trails(background_bgr: np.ndarray, trail: np.ndarray) -> np.ndarray:
    # trail: float32 0..1
    if trail.size == 0:
        return background_bgr
    base = background_bgr.astype(np.float32)
    blue = np.zeros_like(base)
    blue[..., 0] = 255.0
    a = np.clip(trail, 0.0, 1.0)[..., None].astype(np.float32) * 0.75
    out = base * (1.0 - a) + blue * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _track_color(track_id: int) -> Tuple[int, int, int]:
    # Deterministic vivid palette (BGR)
    palette = [
        (255, 80, 80),    # blue-ish
        (80, 255, 80),    # green
        (80, 80, 255),    # red
        (255, 255, 80),   # cyan
        (255, 80, 255),   # magenta
        (80, 255, 255),   # yellow
        (200, 140, 40),   # orange-ish
        (140, 40, 200),   # purple-ish
        (40, 200, 140),   # teal-ish
        (200, 200, 200),  # light gray
    ]
    return palette[track_id % len(palette)]


@dataclass
class _Track:
    track_id: int
    cx: int
    cy: int
    last_seen: int
    color: Tuple[int, int, int]


class PeopleDetector:
    def __init__(self, backend: Literal["yolo", "hog"] = "hog") -> None:
        self.backend = backend
        self._hog = None
        self._yolo = None

        if backend == "hog":
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self._hog = hog
        elif backend == "yolo":
            # Lazy import to keep base install lightweight
            try:
                from ultralytics import YOLO  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "YOLO backend требует `pip install ultralytics` (и зависимости). "
                    f"Ошибка импорта: {e}"
                )
            # Using default lightweight model name; ultralytics will download on first run
            self._yolo = YOLO("yolov8n.pt")
        else:
            raise ValueError("Unknown backend")

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Returns list of (x1,y1,x2,y2,conf) in pixel coords.
        """
        if self.backend == "hog" and self._hog is not None:
            # HOG expects smaller images; keep moderate
            img = frame_bgr
            h, w = img.shape[:2]
            scale = 1.0
            if max(h, w) > 1280:
                scale = 1280.0 / float(max(h, w))
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            rects, weights = self._hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
            out: List[Tuple[int, int, int, int, float]] = []
            for (x, y, ww, hh), conf in zip(rects, weights):
                x1 = int(round(x / scale))
                y1 = int(round(y / scale))
                x2 = int(round((x + ww) / scale))
                y2 = int(round((y + hh) / scale))
                out.append((x1, y1, x2, y2, float(conf)))
            return out

        if self.backend == "yolo" and self._yolo is not None:
            # ultralytics returns xyxy + conf + cls
            res = self._yolo.predict(frame_bgr, verbose=False)[0]
            out: List[Tuple[int, int, int, int, float]] = []
            if res.boxes is None:
                return out
            boxes = res.boxes
            for b in boxes:
                cls = int(b.cls.item()) if b.cls is not None else -1
                # COCO class 0 == person
                if cls != 0:
                    continue
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                conf = float(b.conf.item()) if b.conf is not None else 0.0
                out.append((x1, y1, x2, y2, conf))
            return out

        return []


def process_video(
    video_path: str,
    sample_every_sec: float,
    diff_threshold: int,  # used as MOG2 varThreshold
    bg_alpha: float,  # kept for compatibility; not used in MOG2 mode
    preview_every_updates: int,
    out_queue: Queue,
    output_path: Optional[str] = None,
    stop_event: Optional[threading.Event] = None,
    mode: Literal["once", "heat"] = "once",
    min_area: int = 350,
    blur_ksize: int = 5,
    dilate_iter: int = 2,
    trail_radius: int = 10,
    roi_rel: Optional[Tuple[float, float, float, float]] = None,
    track_dist_px: int = 70,
    track_max_missed: int = 8,
    entry_snapshots: bool = False,
    detector_backend: Literal["yolo", "hog"] = "hog",
    entry_min_conf: float = 0.4,
    entry_target_track_id: int = -1,
) -> None:
    t0 = time.time()
    cap: Optional[cv2.VideoCapture] = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Не удалось открыть видео (cv2.VideoCapture).")

        fps, frame_count = _safe_video_info(cap)
        if fps <= 0:
            fps = 25.0
        step = max(1, int(round(fps * max(0.1, sample_every_sec))))

        ok, first = cap.read()
        if not ok:
            raise RuntimeError("Не удалось прочитать первый кадр видео.")

        h, w = first.shape[:2]
        background_bgr = first.copy()

        # Background subtractor is much better for "one-time approach" visibility
        subtractor = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=max(4, int(diff_threshold)), detectShadows=True)
        subtractor.apply(first)  # warm-up

        detector: Optional[PeopleDetector] = None
        entry_log = None
        entry_state: dict[int, bool] = {}  # track_id -> in_roi
        entry_dir = None
        if entry_snapshots:
            detector = PeopleDetector(detector_backend)
            base_dir = os.path.dirname(output_path) if output_path else os.path.dirname(video_path)
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            entry_dir = os.path.join(base_dir, base_name + "_entries")
            os.makedirs(entry_dir, exist_ok=True)
            entry_log_path = os.path.join(entry_dir, "entries.csv")
            entry_log = open(entry_log_path, "a", encoding="utf-8")
            if entry_log.tell() == 0:
                entry_log.write("video,frame_idx,time_ms,track_id,conf,x1,y1,x2,y2,image_path\n")

        if mode == "heat":
            heat = np.zeros((h, w), dtype=np.float32)  # frequency-ish
            presence = None
        else:
            presence = np.zeros((h, w), dtype=np.uint8)  # OR mask: was motion at least once
            heat = None

        # legacy single-color trail (still used as fallback)
        trail = np.zeros((h, w), dtype=np.float32)  # motion path (centroids)
        # colorized per-employee trail + presence
        trail_color = np.zeros((h, w, 3), dtype=np.uint8)
        presence_color = np.zeros((h, w, 3), dtype=np.uint8)
        tracks: List[_Track] = []
        next_track_id = 0

        roi_px: Optional[Tuple[int, int, int, int]] = None
        roi_mask: Optional[np.ndarray] = None
        if roi_rel is not None:
            x1r, y1r, x2r, y2r = roi_rel
            x1 = int(round(max(0.0, min(1.0, x1r)) * w))
            x2 = int(round(max(0.0, min(1.0, x2r)) * w))
            y1 = int(round(max(0.0, min(1.0, y1r)) * h))
            y2 = int(round(max(0.0, min(1.0, y2r)) * h))
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))
            if (x2 - x1) >= 2 and (y2 - y1) >= 2:
                roi_px = (x1, y1, x2, y2)
                roi_mask = np.zeros((h, w), dtype=np.uint8)
                roi_mask[y1:y2, x1:x2] = 255

        processed = 1
        decoded = 1

        # Estimate total processed frames for progress
        total_est = max(1, (frame_count + step - 1) // step) if frame_count > 0 else None

        last_preview_sent_at = 0
        while True:
            if stop_event is not None and stop_event.is_set():
                out_queue.put(ProgressUpdate(kind="stopped"))
                return
            # Skip step-1 frames without decoding
            for _ in range(step - 1):
                if stop_event is not None and stop_event.is_set():
                    out_queue.put(ProgressUpdate(kind="stopped"))
                    return
                if not cap.grab():
                    break
                decoded += 1
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                decoded += 1
                processed += 1

                background_bgr = frame

                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or decoded)
                time_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)

                fg = subtractor.apply(frame)
                # remove shadows (MOG2 uses 127 for shadows)
                _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

                if blur_ksize and blur_ksize >= 3 and blur_ksize % 2 == 1:
                    fg = cv2.GaussianBlur(fg, (blur_ksize, blur_ksize), 0)
                    _, fg = cv2.threshold(fg, 20, 255, cv2.THRESH_BINARY)

                if dilate_iter > 0:
                    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
                    fg = cv2.dilate(fg, np.ones((3, 3), np.uint8), iterations=int(dilate_iter))

                if roi_mask is not None:
                    fg = cv2.bitwise_and(fg, roi_mask)

                # accumulate motion presence / heat
                if mode == "heat" and heat is not None:
                    heat += (fg.astype(np.float32) / 255.0)
                elif mode == "once" and presence is not None:
                    presence = cv2.bitwise_or(presence, fg)

                # centroid trails (helps see "came once to desk")
                contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                detections: List[Tuple[int, int, np.ndarray]] = []
                for c in contours:
                    area = cv2.contourArea(c)
                    if area < float(min_area):
                        continue
                    m = cv2.moments(c)
                    if m["m00"] <= 1e-6:
                        continue
                    cx = int(m["m10"] / m["m00"])
                    cy = int(m["m01"] / m["m00"])
                    detections.append((cx, cy, c))

                # age tracks
                alive: List[_Track] = []
                for t in tracks:
                    if (processed - t.last_seen) <= int(track_max_missed):
                        alive.append(t)
                tracks = alive

                # greedy assignment: each detection -> nearest track within threshold
                used_tracks = set()
                det_to_track: List[Tuple[int, int, _Track, np.ndarray]] = []
                for cx, cy, c in detections:
                    best_t = None
                    best_d2 = None
                    for t in tracks:
                        if t.track_id in used_tracks:
                            continue
                        dx = cx - t.cx
                        dy = cy - t.cy
                        d2 = dx * dx + dy * dy
                        if best_d2 is None or d2 < best_d2:
                            best_d2 = d2
                            best_t = t
                    if best_t is not None and best_d2 is not None and best_d2 <= int(track_dist_px) * int(track_dist_px):
                        used_tracks.add(best_t.track_id)
                        det_to_track.append((cx, cy, best_t, c))
                    else:
                        # new track
                        tid = next_track_id
                        next_track_id += 1
                        nt = _Track(track_id=tid, cx=cx, cy=cy, last_seen=processed, color=_track_color(tid))
                        tracks.append(nt)
                        used_tracks.add(nt.track_id)
                        det_to_track.append((cx, cy, nt, c))

                # draw per-track marks
                for cx, cy, t, c in det_to_track:
                    t.cx, t.cy, t.last_seen = cx, cy, processed
                    # colored centroid trail
                    cv2.circle(trail_color, (cx, cy), int(max(2, trail_radius)), t.color, thickness=-1)
                    # also keep legacy single-channel trail
                    cv2.circle(trail, (cx, cy), int(max(2, trail_radius)), 1.0, thickness=-1)
                    # colored presence "blob" for this employee
                    cv2.drawContours(presence_color, [c], -1, t.color, thickness=cv2.FILLED)

                # ENTRY SNAPSHOTS: run person detector and save snapshots when entering ROI
                if entry_snapshots and detector is not None and roi_px is not None and entry_log is not None and entry_dir is not None:
                    x1r, y1r, x2r, y2r = roi_px
                    dets = detector.detect(frame)
                    # simple matching between detections and existing tracks by centroid distance
                    for (x1, y1, x2, y2, conf) in dets:
                        if conf < float(entry_min_conf):
                            continue
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        in_roi = (x1r <= cx <= x2r) and (y1r <= cy <= y2r)
                        if not in_roi:
                            continue

                        # find nearest track
                        best_t = None
                        best_d2 = None
                        for t in tracks:
                            dx = cx - t.cx
                            dy = cy - t.cy
                            d2 = dx * dx + dy * dy
                            if best_d2 is None or d2 < best_d2:
                                best_d2 = d2
                                best_t = t
                        if best_t is None:
                            continue
                        if int(entry_target_track_id) >= 0 and best_t.track_id != int(entry_target_track_id):
                            continue

                        was_in = entry_state.get(best_t.track_id, False)
                        if was_in:
                            continue  # already inside

                        entry_state[best_t.track_id] = True
                        snap = frame.copy()
                        cv2.rectangle(snap, (x1r, y1r), (x2r, y2r), (0, 255, 255), 2)
                        cv2.rectangle(snap, (x1, y1), (x2, y2), best_t.color, 2)
                        cv2.putText(
                            snap,
                            f"ID {best_t.track_id} conf {conf:.2f} t={int(time_ms)}ms",
                            (max(5, x1), max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            best_t.color,
                            2,
                            cv2.LINE_AA,
                        )
                        img_name = f"entry_t{int(time_ms):010d}_f{frame_idx:07d}_id{best_t.track_id}.jpg"
                        img_path = os.path.join(entry_dir, img_name)
                        cv2.imwrite(img_path, snap, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                        entry_log.write(
                            f"\"{os.path.basename(video_path)}\",{frame_idx},{time_ms:.0f},{best_t.track_id},{conf:.3f},{x1},{y1},{x2},{y2},\"{img_path}\"\n"
                        )
                        entry_log.flush()

                    # reset state for tracks that are not inside ROI anymore (based on track centroid)
                    for t in tracks:
                        in_now = (x1r <= t.cx <= x2r) and (y1r <= t.cy <= y2r)
                        if not in_now:
                            entry_state[t.track_id] = False

                if processed - last_preview_sent_at >= preview_every_updates:
                    last_preview_sent_at = processed
                    if mode == "heat" and heat is not None:
                        p99 = float(np.percentile(heat, 99.0)) if np.any(heat > 0) else 0.0
                        mask01 = np.clip(heat / max(1e-6, p99), 0.0, 1.0) if p99 > 1e-6 else np.zeros_like(heat)
                        overlay = _overlay_from_mask(background_bgr, mask01)
                    else:
                        mask01 = (presence.astype(np.float32) / 255.0) if presence is not None else np.zeros((h, w), np.float32)
                        overlay = _overlay_from_mask(background_bgr, mask01)
                    # colored employee marks: blobs + trails
                    overlay = cv2.addWeighted(overlay, 1.0, presence_color, 0.55, 0.0)
                    overlay = cv2.addWeighted(overlay, 1.0, trail_color, 0.75, 0.0)
                    overlay = _draw_trails(overlay, np.clip(trail, 0.0, 1.0))
                    if roi_px is not None:
                        x1, y1, x2, y2 = roi_px
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    out_queue.put(
                        ProgressUpdate(
                            kind="preview",
                            preview_bgr=overlay,
                        )
                    )

                # Progress update
                elapsed = time.time() - t0
                if total_est is not None:
                    percent = min(100.0, (processed / float(total_est)) * 100.0)
                    eta = (elapsed / max(1, processed)) * max(0, total_est - processed)
                    out_queue.put(
                        ProgressUpdate(
                            kind="progress",
                            percent=percent,
                            processed=processed,
                            total=total_est,
                            eta_sec=eta,
                        )
                    )
                elif frame_count > 0:
                    pos = cap.get(cv2.CAP_PROP_POS_FRAMES) or decoded
                    percent = min(100.0, (pos / float(frame_count)) * 100.0)
                    out_queue.put(ProgressUpdate(kind="progress", percent=percent))

        if mode == "heat" and heat is not None:
            p99 = float(np.percentile(heat, 99.0)) if np.any(heat > 0) else 0.0
            mask01 = np.clip(heat / max(1e-6, p99), 0.0, 1.0) if p99 > 1e-6 else np.zeros_like(heat)
            overlay = _overlay_from_mask(background_bgr, mask01)
        else:
            mask01 = (presence.astype(np.float32) / 255.0) if presence is not None else np.zeros((h, w), np.float32)
            overlay = _overlay_from_mask(background_bgr, mask01)
        overlay = cv2.addWeighted(overlay, 1.0, presence_color, 0.55, 0.0)
        overlay = cv2.addWeighted(overlay, 1.0, trail_color, 0.75, 0.0)
        overlay = _draw_trails(overlay, np.clip(trail, 0.0, 1.0))
        if roi_px is not None:
            x1, y1, x2, y2 = roi_px
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if not output_path:
            base, _ = os.path.splitext(video_path)
            output_path = base + "_motion_summary.png"
        cv2.imwrite(output_path, overlay)

        out_queue.put(ProgressUpdate(kind="done", output_path=output_path))
    except Exception as e:
        out_queue.put(ProgressUpdate(kind="error", error=str(e)))
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            if entry_log is not None:
                entry_log.close()
        except Exception:
            pass


def _format_eta(sec: Optional[float]) -> str:
    if sec is None or not np.isfinite(sec) or sec < 0:
        return ""
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"ETA {h:02d}:{m:02d}:{s:02d}"
    return f"ETA {m:02d}:{s:02d}"


class App:
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("Сводный кадр (карта движения) из видео")
        self.root.geometry("980x620")

        self.queue: Queue = Queue()
        self.worker: Optional[threading.Thread] = None

        self.video_path = StringVar(value="")
        self.output_dir = StringVar(value="")
        self.auto_next = IntVar(value=1)
        self.sample_every_sec = DoubleVar(value=1.0)
        self.diff_threshold = IntVar(value=25)  # used as MOG2 varThreshold
        self.bg_alpha = DoubleVar(value=0.01)
        self.preview_every = IntVar(value=8)
        self.mode = StringVar(value="once")
        self.min_area = IntVar(value=350)
        self.blur_ksize = IntVar(value=5)
        self.dilate_iter = IntVar(value=2)
        self.trail_radius = IntVar(value=10)
        self.track_dist_px = IntVar(value=70)
        self.track_max_missed = IntVar(value=8)
        self.roi_text = StringVar(value="Зона: вся картинка")
        self._roi_rel: Optional[Tuple[float, float, float, float]] = None
        self.entry_snapshots = IntVar(value=0)
        self.detector_backend = StringVar(value="hog")
        self.entry_min_conf = DoubleVar(value=0.4)
        self.entry_target_track_id = IntVar(value=-1)
        self.suppress_done_popup = IntVar(value=1)

        self.status = StringVar(value="Выберите видео и нажмите «Старт».")

        self._preview_imgtk: Optional[ImageTk.PhotoImage] = None
        self._last_preview_bgr: Optional[np.ndarray] = None
        self._playlist: List[str] = []
        self._current_index: int = -1
        self._stop_event = threading.Event()
        self._auto_next_pending: bool = False
        self._finished_thread: Optional[threading.Thread] = None
        self._auto_next_armed: bool = False
        self._stop_reason: Literal["user", "auto_next"] = "user"
        self._auto_next_triggered_by_progress: bool = False

        self._build_ui()
        self._poll_queue()

    def _build_ui(self) -> None:
        self._build_menu()
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill="both", expand=True)

        top = ttk.Frame(frm)
        top.pack(fill="x")

        ttk.Label(top, text="Видео:").pack(side="left")
        self.path_entry = ttk.Entry(top, textvariable=self.video_path)
        self.path_entry.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(top, text="Выбрать…", command=self.pick_video).pack(side="left")
        ttk.Button(top, text="Добавить в очередь…", command=self.add_videos).pack(side="left", padx=(8, 0))

        outrow = ttk.Frame(frm)
        outrow.pack(fill="x", pady=(8, 0))
        ttk.Label(outrow, text="Папка для сохранения (если пусто — рядом с видео):").pack(side="left")
        ttk.Entry(outrow, textvariable=self.output_dir).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(outrow, text="Выбрать…", command=self.pick_output_dir).pack(side="left")

        opts = ttk.LabelFrame(frm, text="Настройки (для длинных видео лучше больше шаг)", padding=10)
        opts.pack(fill="x", pady=10)

        row1 = ttk.Frame(opts)
        row1.pack(fill="x")
        ttk.Label(row1, text="Обрабатывать кадр каждые (сек):").pack(side="left")
        ttk.Spinbox(row1, from_=0.2, to=60.0, increment=0.2, textvariable=self.sample_every_sec, width=8).pack(
            side="left", padx=8
        )
        ttk.Label(row1, text="Порог движения (0-255):").pack(side="left", padx=(16, 0))
        ttk.Spinbox(row1, from_=1, to=255, increment=1, textvariable=self.diff_threshold, width=6).pack(
            side="left", padx=8
        )

        row2 = ttk.Frame(opts)
        row2.pack(fill="x", pady=(8, 0))
        ttk.Label(row2, text="Режим:").pack(side="left")
        ttk.Combobox(
            row2,
            textvariable=self.mode,
            values=["once", "heat"],
            width=8,
            state="readonly",
        ).pack(side="left", padx=8)
        ttk.Label(row2, text="(once = было хотя бы раз, heat = частота)").pack(side="left")

        row3 = ttk.Frame(opts)
        row3.pack(fill="x", pady=(8, 0))
        ttk.Label(row3, text="Мин. площадь объекта (px):").pack(side="left")
        ttk.Spinbox(row3, from_=50, to=20000, increment=50, textvariable=self.min_area, width=8).pack(
            side="left", padx=8
        )
        ttk.Label(row3, text="Размытие (нечётн., 0=выкл):").pack(side="left", padx=(16, 0))
        ttk.Spinbox(row3, from_=0, to=31, increment=2, textvariable=self.blur_ksize, width=6).pack(side="left", padx=8)
        ttk.Label(row3, text="Дилатация (итерации):").pack(side="left", padx=(16, 0))
        ttk.Spinbox(row3, from_=0, to=10, increment=1, textvariable=self.dilate_iter, width=6).pack(side="left", padx=8)

        row4 = ttk.Frame(opts)
        row4.pack(fill="x", pady=(8, 0))
        ttk.Label(row4, text="Радиус следа (px):").pack(side="left")
        ttk.Spinbox(row4, from_=2, to=40, increment=1, textvariable=self.trail_radius, width=6).pack(
            side="left", padx=8
        )
        ttk.Label(row4, text="Предпросмотр каждые N обработанных кадров:").pack(side="left", padx=(16, 0))
        ttk.Spinbox(row4, from_=1, to=200, increment=1, textvariable=self.preview_every, width=6).pack(
            side="left", padx=8
        )

        row5 = ttk.Frame(opts)
        row5.pack(fill="x", pady=(8, 0))
        ttk.Label(row5, textvariable=self.roi_text).pack(side="left")
        ttk.Button(row5, text="Выбрать зону…", command=self.select_roi).pack(side="left", padx=10)
        ttk.Button(row5, text="Сбросить зону", command=self.reset_roi).pack(side="left")

        row6 = ttk.Frame(opts)
        row6.pack(fill="x", pady=(8, 0))
        ttk.Label(row6, text="Разделение сотрудников: дистанция (px):").pack(side="left")
        ttk.Spinbox(row6, from_=10, to=400, increment=5, textvariable=self.track_dist_px, width=6).pack(
            side="left", padx=8
        )
        ttk.Label(row6, text="пропуск кадров (N):").pack(side="left", padx=(16, 0))
        ttk.Spinbox(row6, from_=1, to=200, increment=1, textvariable=self.track_max_missed, width=6).pack(
            side="left", padx=8
        )

        row7 = ttk.Frame(opts)
        row7.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(row7, text="Фотофиксация входа человека в зону", variable=self.entry_snapshots).pack(side="left")
        ttk.Label(row7, text="Детектор:").pack(side="left", padx=(16, 0))
        ttk.Combobox(row7, textvariable=self.detector_backend, values=["hog", "yolo"], width=6, state="readonly").pack(
            side="left", padx=8
        )
        ttk.Label(row7, text="min conf:").pack(side="left", padx=(16, 0))
        ttk.Spinbox(row7, from_=0.05, to=0.95, increment=0.05, textvariable=self.entry_min_conf, width=6).pack(
            side="left", padx=8
        )

        row8 = ttk.Frame(opts)
        row8.pack(fill="x", pady=(8, 0))
        ttk.Label(row8, text="ID сотрудника для фиксации (-1 = все):").pack(side="left")
        ttk.Spinbox(row8, from_=-1, to=9999, increment=1, textvariable=self.entry_target_track_id, width=8).pack(
            side="left", padx=8
        )
        ttk.Checkbutton(row8, text="Не показывать окно «Готово» (для авто-очереди)", variable=self.suppress_done_popup).pack(
            side="left", padx=16
        )

        actions = ttk.Frame(frm)
        actions.pack(fill="x", pady=(4, 8))
        self.start_btn = ttk.Button(actions, text="Старт", command=self.start)
        self.start_btn.pack(side="left")
        ttk.Checkbutton(actions, text="Следующее видео автоматически", variable=self.auto_next).pack(side="left", padx=10)
        self.progress = ttk.Progressbar(actions, orient="horizontal", mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=10)
        self.stop_btn = ttk.Button(actions, text="Стоп", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left")
        self.save_preview_btn = ttk.Button(actions, text="Сохранить предпросмотр…", command=self.save_preview, state="disabled")
        self.save_preview_btn.pack(side="left", padx=(8, 0))

        mid = ttk.Frame(frm)
        mid.pack(fill="both", expand=True)

        left = ttk.Frame(mid)
        left.pack(side="left", fill="y")

        ttk.Label(left, text="Очередь видео").pack(anchor="w")
        self.playlist_box = ttk.Treeview(left, columns=("path",), show="tree", height=18)
        self.playlist_box.pack(fill="y", expand=True, pady=(4, 0))
        self.playlist_box.bind("<Double-1>", lambda _e: self.start_selected())
        self.playlist_box.bind("<Delete>", lambda _e: self.remove_selected())
        self.playlist_box.bind("<BackSpace>", lambda _e: self.remove_selected())
        self._build_playlist_menu()

        pl_btns = ttk.Frame(left)
        pl_btns.pack(fill="x", pady=(6, 0))
        ttk.Button(pl_btns, text="Удалить", command=self.remove_selected).pack(side="left")
        ttk.Button(pl_btns, text="Очистить", command=self.clear_playlist).pack(side="left", padx=6)
        ttk.Button(pl_btns, text="Следующее", command=self.start_next).pack(side="left", padx=6)
        ttk.Button(pl_btns, text="Старт выбранное", command=self.start_selected).pack(side="left", padx=6)

        self.preview_label = ttk.Label(mid, text="Предпросмотр появится во время обработки.")
        self.preview_label.pack(side="left", fill="both", expand=True, padx=(12, 0))

        bottom = ttk.Frame(frm)
        bottom.pack(fill="x", pady=(8, 0))
        ttk.Label(bottom, textvariable=self.status).pack(side="left")

    def pick_video(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Выберите одно или несколько видео",
            filetypes=[
                ("Видео", "*.mp4 *.avi *.mov *.mkv *.m4v"),
                ("Все файлы", "*.*"),
            ],
        )
        if not paths:
            return
        # "Выбрать…" = заменить очередь выбранными файлами
        self.clear_playlist()
        for p in paths:
            self._add_to_playlist(p)
        if self._playlist:
            self.set_current_video(self._playlist[0], add_to_queue=False)

    def add_videos(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Добавить видео в очередь",
            filetypes=[
                ("Видео", "*.mp4 *.avi *.mov *.mkv *.m4v"),
                ("Все файлы", "*.*"),
            ],
        )
        if not paths:
            return
        for p in paths:
            self._add_to_playlist(p)
        if not self.video_path.get().strip() and self._playlist:
            self.set_current_video(self._playlist[0], add_to_queue=False)

    def pick_output_dir(self) -> None:
        d = filedialog.askdirectory(title="Папка для сохранения картинок")
        if d:
            self.output_dir.set(d)

    def _add_to_playlist(self, path: str) -> None:
        path = os.path.abspath(path)
        if path in self._playlist:
            return
        self._playlist.append(path)
        name = os.path.basename(path)
        self.playlist_box.insert("", "end", iid=path, text=name)

    def set_current_video(self, path: str, add_to_queue: bool) -> None:
        path = os.path.abspath(path)
        self.video_path.set(path)
        if add_to_queue:
            self._add_to_playlist(path)
        if path in self._playlist:
            self._current_index = self._playlist.index(path)
            try:
                self.playlist_box.selection_set(path)
                self.playlist_box.see(path)
            except Exception:
                pass

    def remove_selected(self) -> None:
        sel = self.playlist_box.selection()
        if not sel:
            return
        # If removing current item while processing, stop first.
        cur = os.path.abspath(self.video_path.get().strip()) if self.video_path.get().strip() else None
        if cur and cur in sel and (self.worker and self.worker.is_alive()):
            self.stop()
        for iid in sel:
            if iid in self._playlist:
                idx = self._playlist.index(iid)
                self._playlist.pop(idx)
            try:
                self.playlist_box.delete(iid)
            except Exception:
                pass
        if self._playlist:
            self._current_index = min(self._current_index, len(self._playlist) - 1)
            self.set_current_video(self._playlist[self._current_index], add_to_queue=False)
        else:
            self._current_index = -1
            self.video_path.set("")

    def clear_playlist(self) -> None:
        for p in list(self._playlist):
            try:
                self.playlist_box.delete(p)
            except Exception:
                pass
        self._playlist.clear()
        self._current_index = -1
        self.video_path.set("")

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        path = self.video_path.get().strip()
        if not path:
            if self._playlist:
                self.set_current_video(self._playlist[0], add_to_queue=False)
                path = self.video_path.get().strip()
            else:
                messagebox.showwarning("Видео не выбрано", "Выберите видеофайл или добавьте видео в очередь.")
                return
            return
        if not os.path.exists(path):
            messagebox.showerror("Файл не найден", "Путь к видеофайлу неверный.")
            return

        self.progress["value"] = 0
        self.status.set("Запуск обработки…")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.save_preview_btn.configure(state="normal")
        self._stop_event.clear()
        self._auto_next_pending = False
        self._auto_next_armed = False
        self._auto_next_triggered_by_progress = False
        self._stop_reason = "user"

        # ensure playlist has current item
        self._add_to_playlist(path)
        self._current_index = self._playlist.index(os.path.abspath(path))

        def runner() -> None:
            try:
                out_dir = self.output_dir.get().strip()
                out_path = None
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                    base = os.path.splitext(os.path.basename(path))[0]
                    out_path = os.path.join(out_dir, base + "_motion_summary.png")
                process_video(
                    video_path=path,
                    sample_every_sec=float(self.sample_every_sec.get()),
                    diff_threshold=int(self.diff_threshold.get()),
                    bg_alpha=float(self.bg_alpha.get()),
                    preview_every_updates=int(self.preview_every.get()),
                    out_queue=self.queue,
                    output_path=out_path,
                    stop_event=self._stop_event,
                    mode="once" if self.mode.get().strip().lower() != "heat" else "heat",
                    min_area=int(self.min_area.get()),
                    blur_ksize=int(self.blur_ksize.get()),
                    dilate_iter=int(self.dilate_iter.get()),
                    trail_radius=int(self.trail_radius.get()),
                    roi_rel=self._roi_rel,
                    track_dist_px=int(self.track_dist_px.get()),
                    track_max_missed=int(self.track_max_missed.get()),
                    entry_snapshots=bool(int(self.entry_snapshots.get())),
                    detector_backend="yolo" if self.detector_backend.get().strip().lower() == "yolo" else "hog",
                    entry_min_conf=float(self.entry_min_conf.get()),
                    entry_target_track_id=int(self.entry_target_track_id.get()),
                )
            finally:
                # This is the only reliable signal that the worker thread is fully done.
                self.queue.put(ProgressUpdate(kind="thread_finished"))

        self.worker = threading.Thread(target=runner, daemon=True)
        self.worker.start()

    def stop(self, reason: Literal["user", "auto_next"] = "user") -> None:
        if not (self.worker and self.worker.is_alive()):
            return
        self._stop_reason = reason
        self._stop_event.set()
        self.stop_btn.configure(state="disabled")
        self.status.set("Остановка…")

    def reset_roi(self) -> None:
        self._roi_rel = None
        self.roi_text.set("Зона: вся картинка")

    def select_roi(self) -> None:
        path = self.video_path.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showwarning("Нет видео", "Сначала выберите видео.")
            return

        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            messagebox.showerror("Ошибка", "Не удалось прочитать первый кадр для выбора зоны.")
            return

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        win = Toplevel(self.root)
        win.title("Выделите зону мышкой (прямоугольник)")
        win.geometry("920x680")

        canvas = Canvas(win, bg="black", highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        max_w, max_h = 880, 600
        scale = min(max_w / w, max_h / h, 1.0)
        disp_w = int(w * scale)
        disp_h = int(h * scale)
        disp = img.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(disp)

        offx, offy = 10, 10
        canvas.create_image(offx, offy, anchor="nw", image=imgtk)
        canvas.image = imgtk  # keep ref

        rect_id = None
        start = {"x": 0, "y": 0}

        def clamp_to_image(x: int, y: int) -> Tuple[int, int]:
            x = max(offx, min(offx + disp_w, x))
            y = max(offy, min(offy + disp_h, y))
            return x, y

        def on_down(evt) -> None:
            nonlocal rect_id
            x, y = clamp_to_image(evt.x, evt.y)
            start["x"], start["y"] = x, y
            if rect_id is not None:
                canvas.delete(rect_id)
            rect_id = canvas.create_rectangle(x, y, x, y, outline="yellow", width=2)

        def on_move(evt) -> None:
            nonlocal rect_id
            if rect_id is None:
                return
            x, y = clamp_to_image(evt.x, evt.y)
            canvas.coords(rect_id, start["x"], start["y"], x, y)

        def on_up(evt) -> None:
            nonlocal rect_id
            if rect_id is None:
                return
            x, y = clamp_to_image(evt.x, evt.y)
            x1, y1 = start["x"], start["y"]
            x2, y2 = x, y
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))

            x1o = int(round((x1 - offx) / scale))
            y1o = int(round((y1 - offy) / scale))
            x2o = int(round((x2 - offx) / scale))
            y2o = int(round((y2 - offy) / scale))

            if (x2o - x1o) < 10 or (y2o - y1o) < 10:
                return

            self._roi_rel = (x1o / w, y1o / h, x2o / w, y2o / h)
            self.roi_text.set(f"Зона: x {x1o}-{x2o}, y {y1o}-{y2o}")

        canvas.bind("<Button-1>", on_down)
        canvas.bind("<B1-Motion>", on_move)
        canvas.bind("<ButtonRelease-1>", on_up)

        btns = ttk.Frame(win, padding=8)
        btns.pack(fill="x")

        ttk.Button(btns, text="Сбросить", command=self.reset_roi).pack(side="left")
        ttk.Button(btns, text="Готово", command=win.destroy).pack(side="right")

    def start_next(self) -> None:
        if self.worker and self.worker.is_alive():
            # wait a bit and retry (thread may still be finishing)
            self.root.after(250, self.start_next)
            return
        if not self._playlist:
            messagebox.showwarning("Очередь пуста", "Добавьте видео в очередь.")
            return
        nxt = 0 if self._current_index < 0 else self._current_index + 1
        if nxt >= len(self._playlist):
            self.status.set("Очередь закончилась.")
            if int(self.suppress_done_popup.get()) != 1:
                messagebox.showinfo("Готово", "Очередь закончилась.")
            return
        self.set_current_video(self._playlist[nxt], add_to_queue=False)
        self.start()

    def start_selected(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        sel = self.playlist_box.selection()
        if not sel:
            messagebox.showwarning("Не выбрано", "Выберите видео в очереди.")
            return
        path = sel[0]
        if path not in self._playlist:
            return
        self.set_current_video(path, add_to_queue=False)
        self.start()

    def _auto_advance_if_needed(self) -> None:
        if int(self.auto_next.get()) != 1:
            return
        if not self._playlist:
            return
        if self._auto_next_pending:
            return
        self._auto_next_pending = True
        # Actual start is triggered on "thread_finished"; here we just arm it.
        self._auto_next_armed = True

    def _auto_advance_tick(self) -> None:
        # kept for backward compatibility; no longer used
        return

    def _set_preview(self, bgr: np.ndarray) -> None:
        self._last_preview_bgr = bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        # Fit into window area
        max_w = max(320, self.preview_label.winfo_width() - 10)
        max_h = max(240, self.preview_label.winfo_height() - 10)
        img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self._preview_imgtk = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self._preview_imgtk, text="")

    def save_preview(self) -> None:
        if self._last_preview_bgr is None:
            messagebox.showwarning("Нет предпросмотра", "Предпросмотр появится после начала обработки.")
            return
        initial = ""
        cur = self.video_path.get().strip()
        if cur:
            base = os.path.splitext(os.path.basename(cur))[0]
            initial = base + "_preview.png"
        path = filedialog.asksaveasfilename(
            title="Сохранить предпросмотр",
            defaultextension=".png",
            initialfile=initial or "preview.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("Все файлы", "*.*")],
        )
        if not path:
            return
        bgr = self._last_preview_bgr
        ext = os.path.splitext(path)[1].lower()
        if ext in [".jpg", ".jpeg"]:
            cv2.imwrite(path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        else:
            cv2.imwrite(path, bgr)
        messagebox.showinfo("Сохранено", f"Сохранено:\n{path}")

    def _poll_queue(self) -> None:
        try:
            while True:
                upd: ProgressUpdate = self.queue.get_nowait()
                if upd.kind == "progress" and upd.percent is not None:
                    self.progress["value"] = upd.percent
                    eta_txt = _format_eta(upd.eta_sec)
                    if upd.processed is not None and upd.total is not None:
                        self.status.set(f"{upd.percent:5.1f}%  ({upd.processed}/{upd.total})  {eta_txt}".strip())
                    else:
                        self.status.set(f"{upd.percent:5.1f}%  {eta_txt}".strip())

                    # If progress reaches the end but "done" doesn't arrive (e.g. estimate mismatch),
                    # force-stop and advance to the next video automatically.
                    if (
                        float(upd.percent) >= 99.9
                        and int(self.auto_next.get()) == 1
                        and not self._auto_next_triggered_by_progress
                        and self._playlist
                        and (self._current_index + 1) < len(self._playlist)
                    ):
                        self._auto_next_triggered_by_progress = True
                        self._auto_next_pending = True
                        self._auto_next_armed = True
                        self.stop(reason="auto_next")
                elif upd.kind == "preview" and upd.preview_bgr is not None:
                    self._set_preview(upd.preview_bgr)
                elif upd.kind == "done":
                    self.progress["value"] = 100
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    self.save_preview_btn.configure(state="disabled")
                    self._finished_thread = self.worker
                    outp = upd.output_path or ""
                    self.status.set(f"Готово. Сохранено: {outp}")
                    # IMPORTANT: messagebox blocks the UI thread and breaks "auto next".
                    # So: auto-advance first, and optionally suppress popups in queue mode.
                    self._auto_advance_if_needed()
                    if outp and int(self.suppress_done_popup.get()) != 1 and int(self.auto_next.get()) != 1:
                        messagebox.showinfo("Готово", f"Сохранено:\n{outp}")
                elif upd.kind == "stopped":
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    self.save_preview_btn.configure(state="disabled")
                    self._finished_thread = self.worker
                    # If stop was triggered for auto-next, keep auto-next armed.
                    if self._stop_reason != "auto_next":
                        self._auto_next_armed = False
                    self.status.set("Остановлено пользователем.")
                elif upd.kind == "error":
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    self.save_preview_btn.configure(state="disabled")
                    self._finished_thread = self.worker
                    self._auto_next_armed = False
                    self.status.set("Ошибка обработки.")
                    messagebox.showerror("Ошибка", upd.error or "Неизвестная ошибка")
                elif upd.kind == "thread_finished":
                    # worker thread is fully stopped; now we can safely start next
                    self.worker = None
                    self._finished_thread = None
                    if self._auto_next_pending and self._auto_next_armed and int(self.auto_next.get()) == 1:
                        self._auto_next_pending = False
                        self._auto_next_armed = False
                        self.start_next()
                    else:
                        self._auto_next_pending = False
        except Empty:
            pass
        self.root.after(120, self._poll_queue)

    def _build_menu(self) -> None:
        menubar = Menu(self.root)
        helpm = Menu(menubar, tearoff=0)
        helpm.add_command(label="Как определить подход к объекту", command=self.show_help_printer)
        helpm.add_command(label="Что означает каждая настройка", command=self.show_help_settings)
        helpm.add_separator()
        helpm.add_command(label="О программе", command=self.show_about)
        menubar.add_cascade(label="Справка", menu=helpm)
        self.root.config(menu=menubar)

    def _build_playlist_menu(self) -> None:
        menu = Menu(self.root, tearoff=0)
        menu.add_command(label="Старт выбранное", command=self.start_selected)
        menu.add_command(label="Удалить", command=self.remove_selected)

        def popup(evt) -> None:
            try:
                iid = self.playlist_box.identify_row(evt.y)
                if iid:
                    self.playlist_box.selection_set(iid)
                menu.tk_popup(evt.x_root, evt.y_root)
            finally:
                try:
                    menu.grab_release()
                except Exception:
                    pass

        self.playlist_box.bind("<Button-3>", popup)

    def _show_text_window(self, title: str, text: str) -> None:
        win = Toplevel(self.root)
        win.title(title)
        win.geometry("860x560")
        frm = ttk.Frame(win, padding=12)
        frm.pack(fill="both", expand=True)
        t = ttk.Treeview  # silence some linters about Text; we use Label with wrap for simplicity
        lbl = ttk.Label(frm, text=text, justify="left", wraplength=820)
        lbl.pack(fill="both", expand=True)
        ttk.Button(frm, text="Закрыть", command=win.destroy).pack(anchor="e", pady=(10, 0))

    def show_help_printer(self) -> None:
        self._show_text_window(
            "Как определить подход к объекту",
            "1) Нажмите «Выбрать видео».\n"
            "2) Нажмите «Выбрать зону…» и выделите прямоугольник вокруг объекта (как на вашем скриншоте).\n"
            "3) Включите «Фотофиксация входа человека в зону».\n"
            "4) Детектор:\n"
            "   - hog: без установки, средняя точность.\n"
            "   - yolo: точнее (нужно установить ultralytics).\n"
            "5) Нажмите «Старт».\n\n"
            "Результат:\n"
            "- В папке рядом с видео появится «<имя>_entries».\n"
            "- Там будут фото каждого события входа + файл entries.csv.\n\n"
            "Как выделить конкретного сотрудника:\n"
            "- На фото входа подписан ID (например, ID 3).\n"
            "- Укажите этот ID в поле «ID сотрудника для фиксации» и повторите обработку — сохранит только этого человека.\n",
        )

    def show_help_settings(self) -> None:
        self._show_text_window(
            "Что означает каждая настройка",
            "Обрабатывать кадр каждые (сек):\n"
            "- Меньше = точнее события (вход/подход), но медленнее.\n\n"
            "Порог движения (varThreshold для MOG2):\n"
            "- Больше = меньше реагирует на шум/тени, но может пропускать движение.\n\n"
            "Мин. площадь объекта:\n"
            "- Отсекает мелкий шум. Если человека не видит — уменьшайте.\n\n"
            "Размытие / Дилатация:\n"
            "- Убирают шум и делают силуэт цельнее.\n\n"
            "Режим once/heat:\n"
            "- once: где было движение хотя бы раз.\n"
            "- heat: где движение было часто.\n\n"
            "Разделение сотрудников (дистанция/пропуск):\n"
            "- Дистанция: насколько близко считается тем же человеком.\n"
            "- Пропуск: сколько кадров трек живёт без обнаружения.\n\n"
            "Фотофиксация входа:\n"
            "- Срабатывает, когда центр bbox человека впервые попадает в ROI.\n"
            "- min conf: порог уверенности детектора (для YOLO).\n",
        )

    def show_about(self) -> None:
        self._show_text_window(
            "О программе",
            "Инструмент для быстрого анализа длинных CCTV-видео.\n"
            "Умеет строить сводную картинку движения, выделять движение в зоне и фиксировать события входа человека в выбранную область.\n",
        )

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    App().run()


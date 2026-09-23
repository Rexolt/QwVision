
import math
import cv2
import numpy as np
import os
from functools import lru_cache
import time
from config import (SAVE_VIDEO, OUTPUT_VIDEO_PATH, ENABLE_SOUND_ALERT, ALERT_SOUND_PATH,
                    TARGET_COLOR, RAW_VIDEO_DIR)

TARGET_DRAW_COLOR = (0, 0, 255)  # BGR piros: a fekete golyón a fekete jelölés nem látszana


class VideoSaver:
    def __init__(self, frame_width, frame_height, fps):
        self.out = None
        if SAVE_VIDEO:

            directory = os.path.dirname(OUTPUT_VIDEO_PATH)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    def write(self, frame):
        if self.out:
            self.out.write(frame)

    def release(self):
        if self.out:
            self.out.release()


class RawRecorder:
    """Jelölések nélküli felvétel, amit CAMERA_SOURCE-ként visszajátszva ugyanaz a jelenet újra hangolható."""

    def __init__(self, frame_width, frame_height, fps):
        self.size = (frame_width, frame_height)
        self.fps = fps
        self.out = None
        self.path = None

    @property
    def recording(self):
        return self.out is not None

    def toggle(self):
        if self.out:
            self.release()
            print("Felvétel mentve:", self.path)
            return
        os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
        self.path = os.path.join(RAW_VIDEO_DIR, time.strftime("raw_%Y%m%d_%H%M%S.avi"))
        # MJPG: nagyobb fájl, de kevésbé torzít, mint az XVID, így jobban hasonlít az élő képre
        self.out = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"MJPG"), self.fps, self.size)
        print("Felvétel indul:", self.path)

    def write(self, frame):
        if self.out:
            self.out.write(frame)

    def release(self):
        if self.out:
            self.out.release()
            self.out = None


@lru_cache(maxsize=1)
def _load_alert_sound():
    import simpleaudio as sa
    return sa.WaveObject.from_wave_file(ALERT_SOUND_PATH)


def play_sound_alert():
    if ENABLE_SOUND_ALERT:
        try:
            _load_alert_sound().play()
        except Exception as e:
            print("Hiba a hang lejátszása közben ellenőrizze a fájl PATH-et:", e)


def track_color(track_id):
    # Azonosítónként eltérő, de futásról futásra azonos szín
    hue = (track_id * 47) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in bgr)


def draw_dashed_polyline(frame, points, color, thickness=1, dash=10, gap=7):
    drawing, remaining = True, dash
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        length = math.hypot(x1 - x0, y1 - y0)
        pos = 0.0
        while pos < length:
            step = min(remaining, length - pos)
            if drawing:
                a, b = pos / length, (pos + step) / length
                cv2.line(frame, (int(x0 + (x1 - x0) * a), int(y0 + (y1 - y0) * a)),
                         (int(x0 + (x1 - x0) * b), int(y0 + (y1 - y0) * b)), color, thickness, cv2.LINE_AA)
            pos += step
            remaining -= step
            if remaining <= 0:
                drawing = not drawing
                remaining = dash if drawing else gap


def _path_hits(path, radius, target):
    """Elhalad-e a jósolt pálya olyan közel a célhoz, hogy a két golyó összeérjen."""
    cx, cy = target.position
    reach = target.radius + radius
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        dx, dy = x1 - x0, y1 - y0
        t = max(0.0, min(1.0, ((cx - x0) * dx + (cy - y0) * dy) / (dx * dx + dy * dy or 1.0)))
        if math.hypot(x0 + dx * t - cx, y0 + dy * t - cy) <= reach:
            return True
    return False


def draw_tracks(frame, tracks, fps, process_ms):
    height, width = frame.shape[:2]
    targets = [t for t in tracks if t.color == TARGET_COLOR]

    for track in tracks:
        is_target = track.color == TARGET_COLOR
        color = TARGET_DRAW_COLOR if is_target else track_color(track.id)
        x, y = track.position
        center = (int(x), int(y))
        radius = int(track.radius)
        # Ha épp nem látjuk, csak a becsült helyét rajzoljuk, vékonyabban
        thickness = 2 if track.missed == 0 else 1

        if len(track.trail) > 1:
            cv2.polylines(frame, [np.array(track.trail, np.int32)], False, color, 1, cv2.LINE_AA)

        # Jósolt pálya szaggatottan; ha a cél golyót is eltalálná, cél színnel
        hits_target = False
        path = track.predicted_path(width, height)
        if path and len(path) > 1:
            hits_target = any(_path_hits(path, track.radius, t) for t in targets if t is not track)
            draw_dashed_polyline(frame, path, TARGET_DRAW_COLOR if hits_target else color, 2)

        cv2.circle(frame, center, radius, color, thickness, cv2.LINE_AA)
        if is_target:
            d = int(radius * 0.7)
            cv2.line(frame, (center[0] - d, center[1] - d), (center[0] + d, center[1] + d), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (center[0] - d, center[1] + d), (center[0] + d, center[1] - d), color, thickness, cv2.LINE_AA)
        else:
            cv2.circle(frame, center, 3, color, -1, cv2.LINE_AA)

        label = f"#{track.id} {track.color}"
        if is_target:
            label += " (CEL)"
        elif hits_target:
            label += " -> CEL"
        label_y = max(int(y - radius - 8), 15)
        cv2.putText(frame, label, (int(x - radius), label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, f"FPS: {fps:.1f}  feldolgozas: {process_ms:.1f} ms", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Golyok: {len(tracks)}", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return frame

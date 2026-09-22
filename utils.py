
import cv2
import numpy as np
import os
from functools import lru_cache
from config import SAVE_VIDEO, OUTPUT_VIDEO_PATH, ENABLE_SOUND_ALERT, ALERT_SOUND_PATH


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


def draw_tracks(frame, tracks, fps, process_ms):
    for track in tracks:
        color = track_color(track.id)
        x, y = track.position
        center = (int(x), int(y))
        radius = int(track.radius)
        # Ha épp nem látjuk, csak a becsült helyét rajzoljuk, vékonyabban
        thickness = 2 if track.missed == 0 else 1

        if len(track.trail) > 1:
            cv2.polylines(frame, [np.array(track.trail, np.int32)], False, color, 1, cv2.LINE_AA)
        cv2.circle(frame, center, radius, color, thickness, cv2.LINE_AA)
        cv2.circle(frame, center, 3, color, -1, cv2.LINE_AA)

        # Sebességvektor: hová ér 5 képkocka múlva
        vx, vy = track.velocity
        tip = (int(x + vx * 5), int(y + vy * 5))
        if tip != center:
            cv2.arrowedLine(frame, center, tip, color, 2, cv2.LINE_AA, tipLength=0.3)

        label_y = max(int(y - radius - 8), 15)
        cv2.putText(frame, f"#{track.id}", (int(x - radius), label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, f"FPS: {fps:.1f}  feldolgozas: {process_ms:.1f} ms", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Korok: {len(tracks)}", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return frame


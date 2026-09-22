
import os
# Az opencv-python Qt-je nem tartalmaz Wayland plugint, ezért X11-en (XWayland) nyitjuk az ablakot
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import signal
import time
from camera import CameraStream
from detection import CircleDetector
from tracker import CircleTracker
from utils import draw_tracks, VideoSaver, play_sound_alert
from config import CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, ALERT_COOLDOWN

WINDOW_NAME = "Kor kovetes"

def _handle_sigterm(signum, stack):
    # Külső leállításkor (kill) is fusson le a takarítás, különben a videófájl olvashatatlan marad
    raise KeyboardInterrupt

def main():
    signal.signal(signal.SIGTERM, _handle_sigterm)
    
    camera = CameraStream(CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
    if not camera.start():
        print(f"Nem sikerült megnyitni a kamerát (forrás: {CAMERA_SOURCE})!")
        print("Iriun Webcam esetén indítsd el a számítógépes és a telefonos appot is.")
        camera.stop()
        return

    detector = CircleDetector()
    tracker = CircleTracker()

    frame, frame_id = camera.frame, camera.frame_id
    frame_height, frame_width = frame.shape[:2]
    video_saver = VideoSaver(frame_width, frame_height, camera.fps or CAMERA_FPS)

    last_alert_time = 0.0
    fps = 0.0
    prev_time = time.perf_counter()

    try:
        while True:
            start = time.perf_counter()
            detections = detector.detect(frame)
            new_tracks = tracker.update(detections)
            process_ms = (time.perf_counter() - start) * 1000

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6) if fps else 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            if new_tracks and now - last_alert_time > ALERT_COOLDOWN:
                play_sound_alert()
                last_alert_time = now

            draw_tracks(frame, tracker.visible_tracks(), fps, process_ms)
            video_saver.write(frame)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # q vagy Esc
                break

            frame, frame_id = camera.read(frame_id)
            if frame is None:
                print("A kamera nem ad több képkockát, kilépés.")
                break
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        camera.stop()
        video_saver.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

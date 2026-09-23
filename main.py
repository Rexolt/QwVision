
import os
# Az opencv-python Qt-je nem tartalmaz Wayland plugint, ezért X11-en (XWayland) nyitjuk az ablakot
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import signal
import time
from camera import CameraStream
from detection import CircleDetector
from tracker import CircleTracker
from utils import draw_tracks, VideoSaver, RawRecorder, play_sound_alert
from config import CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, ALERT_COOLDOWN, COLOR_REDETECT

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

    frame, frame_id, timestamp = camera.frame, camera.frame_id, camera.timestamp
    frame_height, frame_width = frame.shape[:2]
    video_fps = camera.fps or CAMERA_FPS
    video_saver = VideoSaver(frame_width, frame_height, video_fps)
    raw_recorder = RawRecorder(frame_width, frame_height, video_fps)
    # Videófájlt nagyjából valós sebességgel játszunk vissza; élő kamerát nem kell visszafogni
    frame_delay = max(1, int(1000 / video_fps)) if camera.is_file else 1

    print("Gombok: szóköz = szünet, n = következő képkocka (szünetben), r = nyers felvétel be/ki, q / Esc = kilépés")

    last_alert_time = 0.0
    fps = 0.0
    prev_time = time.perf_counter()
    paused = False

    try:
        while True:
            if frame is not None:
                start = time.perf_counter()
                detections, colors = detector.detect(frame)
                current = frame
                redetect = (lambda x, y, r, color: detector.detect_near(current, x, y, r, color)) if COLOR_REDETECT else None
                new_tracks = tracker.update(detections, colors, timestamp, redetect)
                process_ms = (time.perf_counter() - start) * 1000

                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6) if fps else 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                if new_tracks and now - last_alert_time > ALERT_COOLDOWN:
                    play_sound_alert()
                    last_alert_time = now

                # A nyers felvétel a jelölések felrajzolása előtt készül
                raw_recorder.write(frame)
                draw_tracks(frame, tracker.visible_tracks(), fps, process_ms)
                if raw_recorder.recording:
                    cv2.circle(frame, (frame_width - 20, 20), 8, (0, 0, 255), -1)
                video_saver.write(frame)
                cv2.imshow(WINDOW_NAME, frame)
                frame = None

            key = cv2.waitKey(30 if paused else frame_delay) & 0xFF
            if key in (ord('q'), 27):  # q vagy Esc
                break
            if key == ord(' '):
                paused = not paused
            elif key == ord('r'):
                raw_recorder.toggle()
            if paused and key != ord('n'):
                continue

            frame, frame_id, timestamp = camera.read(frame_id)
            if frame is None:
                print("A kamera nem ad több képkockát, kilépés.")
                break
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        camera.stop()
        video_saver.release()
        raw_recorder.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

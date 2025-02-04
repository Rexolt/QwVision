
import cv2
import time
import os
from config import SAVE_VIDEO, OUTPUT_VIDEO_PATH, FRAME_RATE, ENABLE_SOUND_ALERT, ALERT_SOUND_PATH, ALERT_COOLDOWN


class VideoSaver:
    def __init__(self, frame_width, frame_height):
        self.out = None
        if SAVE_VIDEO:
           
            directory = os.path.dirname(OUTPUT_VIDEO_PATH)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FRAME_RATE, (frame_width, frame_height))
    
    def write(self, frame):
        if self.out:
            self.out.write(frame)
    
    def release(self):
        if self.out:
            self.out.release()


def play_sound_alert():
    if ENABLE_SOUND_ALERT:
        try:
            
            import simpleaudio as sa
            wave_obj = sa.WaveObject.from_wave_file(ALERT_SOUND_PATH)
            play_obj = wave_obj.play()
          
        except Exception as e:
            print("Hiba a hang lejátszása közben:", e)


def draw_detections(frame, detections, stats, fps):
   
    for det in detections:
        x, y, w, h = det["box"]
        label = det["label"]
        confidence = det["confidence"]
        color = [int(c) for c in det["color"]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    
    y_offset = 40
    for label, count in stats.items():
        cv2.putText(frame, f"{label}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 20

    return frame


def get_detection_stats(detections):
    stats = {}
    for det in detections:
        label = det["label"]
        stats[label] = stats.get(label, 0) + 1
    return stats

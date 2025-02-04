
import cv2
import time
from detection import ObjectDetector
from utils import draw_detections, get_detection_stats, VideoSaver, play_sound_alert
from config import ENABLE_SOUND_ALERT, ALERT_COOLDOWN

def main():
    detector = ObjectDetector()
    cap = cv2.VideoCapture(0)
    
    
    ret, frame = cap.read()
    if not ret:
        print("Nem sikerült megnyitni a webkamerát!")
        return
    
    frame_height, frame_width = frame.shape[:2]
    video_saver = VideoSaver(frame_width, frame_height)
    
  
    last_alert_time = {}
    
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
       
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        
     
        detections = detector.detect(frame)
        
       
        stats = get_detection_stats(detections)
        
       
        for det in detections:
            if det["label"] == "person":
                now = time.time()
                if "person" not in last_alert_time or (now - last_alert_time["person"] > ALERT_COOLDOWN):
                    play_sound_alert()
                    last_alert_time["person"] = now
        
        
        frame = draw_detections(frame, detections, stats, fps)
        
        
        video_saver.write(frame)
        
        cv2.imshow("Objektum Felismeres", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    video_saver.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

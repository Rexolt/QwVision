

# Model fájlok elérési útjai
WEIGHTS_PATH = "models/yolov3.weights"
CONFIG_PATH = "models/yolov3.cfg"
NAMES_PATH = "models/coco.names"


CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4


BLOB_SCALE = 1/255.0
BLOB_SIZE = (416, 416)
BLOB_SWAP_RB = True


SAVE_VIDEO = True
OUTPUT_VIDEO_PATH = "output/detections.avi"
FRAME_RATE = 20.0


ENABLE_SOUND_ALERT = True
ALERT_SOUND_PATH = "sounds/alert.wav"  
ALERT_COOLDOWN = 5  


import cv2
import numpy as np
from config import WEIGHTS_PATH, CONFIG_PATH, NAMES_PATH, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, BLOB_SCALE, BLOB_SIZE, BLOB_SWAP_RB

class ObjectDetector:
    def __init__(self):
       
        with open(NAMES_PATH, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
       
        self.net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)
        
        
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
        
       
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
    
    def detect(self, frame):
        height, width, channels = frame.shape
        
        blob = cv2.dnn.blobFromImage(frame, scalefactor=BLOB_SCALE, size=BLOB_SIZE, swapRB=BLOB_SWAP_RB, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        class_ids = []
        confidences = []
        boxes = []
        
      
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > CONFIDENCE_THRESHOLD:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        
        detections = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                detections.append({
                    "box": (x, y, w, h),
                    "confidence": confidences[i],
                    "class_id": class_ids[i],
                    "label": self.classes[class_ids[i]],
                    "color": self.colors[class_ids[i]].tolist()
                })
        return detections


import cv2
import numpy as np
from config import PROCESS_WIDTH, BLUR_KERNEL, HOUGH_DP, HOUGH_PARAM1, HOUGH_PARAM2, MIN_RADIUS, MAX_RADIUS, MIN_DIST, MAX_BRIGHTNESS

class CircleDetector:
    def detect(self, frame):
        """Fekete / közel fekete kör alakú objektumok keresése. Visszatérés: (N, 3) tömb (x, y, r), eredeti képkoordinátákban."""
        width = frame.shape[1]
        scale = min(1.0, PROCESS_WIDTH / width)

        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT_ALT,
            dp=HOUGH_DP,
            minDist=max(1.0, MIN_DIST * scale),
            param1=HOUGH_PARAM1,
            param2=HOUGH_PARAM2,
            minRadius=max(1, int(MIN_RADIUS * scale)),
            maxRadius=int(MAX_RADIUS * scale),
        )

        if circles is None:
            return np.empty((0, 3), np.float32)
        circles = circles[0]
        if MAX_BRIGHTNESS < 255:
            # HSV V csatorna = a három színcsatorna maximuma
            value = frame.max(axis=2)
            circles = circles[np.array([self._is_dark(value, c) for c in circles], bool)]
        return circles / scale
    
    @staticmethod
    def _is_dark(value, circle):
        x, y, r = circle
        # Csak a belső részt nézzük, hogy a széle és a háttér ne számítson bele
        inner = max(1.0, r * 0.7)
        x0, y0 = max(0, int(x - inner)), max(0, int(y - inner))
        patch = value[y0:int(y + inner) + 1, x0:int(x + inner) + 1]
        if patch.size == 0:
            return False
        yy, xx = np.ogrid[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= inner ** 2
        pixels = patch[mask]
        # Medián: a fényes csillanás egy fekete labdán nem rontja el
        return pixels.size > 0 and np.median(pixels) <= MAX_BRIGHTNESS

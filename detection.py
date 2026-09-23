
import cv2
import numpy as np
from config import (PROCESS_WIDTH, BLUR_KERNEL, HOUGH_DP, HOUGH_PARAM1, HOUGH_PARAM2, MIN_RADIUS, MAX_RADIUS, MIN_DIST,
                    BLACK_MAX_VALUE, GRAY_MAX_SATURATION, WHITE_MIN_VALUE, TRACKED_COLORS,
                    MAX_MATCH_DISTANCE, REDETECT_MIN_FILL, REDETECT_MAX_FILL, DUPLICATE_OVERLAP,
                    BACKGROUND_BRIGHTNESS_RANGE, BACKGROUND_MAX_COLOR_DIFF)

# OpenCV árnyalat (0..180) felső határai; ami a végén kimarad, az újra piros
HUE_NAMES = [(8, "piros"), (22, "narancs"), (35, "sarga"), (85, "zold"),
             (100, "turkiz"), (130, "kek"), (150, "lila"), (170, "rozsaszin")]
HUE_RANGES = {name: (low, high) for (low, _), (high, name) in zip([(0, None)] + HUE_NAMES, HUE_NAMES)}
COLOR_NAMES = ["fekete", "feher", "szurke"] + [name for _, name in HUE_NAMES]


def color_mask(hsv, name):
    """Azok a pixelek (HSV kép), amelyek az adott színnévbe tartoznak."""
    h, s, v = cv2.split(hsv)
    dark = v <= BLACK_MAX_VALUE
    if name == "fekete":
        return dark
    pale = ~dark & (s <= GRAY_MAX_SATURATION)
    if name == "feher":
        return pale & (v >= WHITE_MIN_VALUE)
    if name == "szurke":
        return pale & (v < WHITE_MIN_VALUE)
    colorful = ~dark & ~pale
    if name == "piros":
        return colorful & ((h < HUE_NAMES[0][0]) | (h >= HUE_NAMES[-1][0]))
    low, high = HUE_RANGES[name]
    return colorful & (h >= low) & (h < high)


def classify_color(bgr):
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)
    return next(name for name in COLOR_NAMES if color_mask(hsv, name)[0, 0])


class CircleDetector:
    def detect(self, frame):
        """Kör alakú objektumok keresése és színük meghatározása.
        Visszatérés: ((N, 3) tömb (x, y, r) eredeti képkoordinátákban, N elemű színnév-lista)."""
        width = frame.shape[1]
        scale = min(1.0, PROCESS_WIDTH / width)

        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(frame, (BLUR_KERNEL, BLUR_KERNEL), 0)
        min_dist = max(1.0, MIN_DIST * scale)

        # Színcsatornánként keresünk: egy piros golyó a zöld posztón szürkeárnyalatosan
        # ugyanolyan fényes lehet, mint a háttér, de a piros csatornán éles a széle
        found = []
        for channel in cv2.split(blurred):
            circles = cv2.HoughCircles(
                channel, cv2.HOUGH_GRADIENT_ALT,
                dp=HOUGH_DP,
                minDist=min_dist,
                param1=HOUGH_PARAM1,
                param2=HOUGH_PARAM2,
                minRadius=max(1, int(MIN_RADIUS * scale)),
                maxRadius=int(MAX_RADIUS * scale),
            )
            if circles is not None:
                found.extend(circles[0])

        # Előbb a színt nézzük (és kiszűrjük az árnyékot / padlót), csak utána a duplikátumokat,
        # különben egy árnyék-kör kiszoríthatná a mellette lévő igazi golyót
        circles, colors = [], []
        for c in found:
            name = self._color_of(frame, c)
            if name is None or (TRACKED_COLORS is not None and name not in TRACKED_COLORS):
                continue
            # Ugyanazt a golyót több csatorna is megtalálhatja: ha egy kör középpontja egy már
            # megtartott körbe esik, az ugyanaz a golyó (két érintkező golyó középpontja messzebb van)
            if any(np.hypot(c[0] - k[0], c[1] - k[1]) < DUPLICATE_OVERLAP * max(c[2], k[2]) for k in circles):
                continue
            circles.append(c)
            colors.append(name)
        if not circles:
            return np.empty((0, 3), np.float32), []
        return np.array(circles, np.float32) / scale, colors

    @staticmethod
    def detect_near(frame, x, y, r, color):
        """A megadott színű golyó keresése színes foltként (x, y) környékén, a körkeresés tartalékaként.
        Visszatérés: (x, y, r) vagy None."""
        height, width = frame.shape[:2]
        reach = r + MAX_MATCH_DISTANCE
        x0, y0 = max(0, int(x - reach)), max(0, int(y - reach))
        x1, y1 = min(width, int(x + reach) + 1), min(height, int(y + reach) + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        mask = cv2.morphologyEx(color_mask(hsv, color).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)

        expected = np.pi * r * r
        best, best_dist = None, MAX_MATCH_DISTANCE
        for i in range(1, count):
            if not REDETECT_MIN_FILL * expected <= stats[i, cv2.CC_STAT_AREA] <= REDETECT_MAX_FILL * expected:
                continue
            cx, cy = centroids[i][0] + x0, centroids[i][1] + y0
            dist = np.hypot(cx - x, cy - y)
            if dist <= best_dist and not CircleDetector._looks_like_background(frame, cx, cy, r):
                best, best_dist = (cx, cy, r), dist
        return best

    @staticmethod
    def _median_color(frame, x, y, r_min, r_max):
        """A (x, y) körüli, r_min..r_max sugarú gyűrű (r_min=0: korong) csatornánkénti medián színe, vagy None."""
        x0, y0 = max(0, int(x - r_max)), max(0, int(y - r_max))
        patch = frame[y0:int(y + r_max) + 1, x0:int(x + r_max) + 1]
        if patch.size == 0:
            return None
        yy, xx = np.ogrid[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]]
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        pixels = patch[(d2 >= r_min ** 2) & (d2 <= r_max ** 2)]
        # Medián: a fényes csillanás, ill. a gyűrűbe belógó szomszéd golyó nem rontja el
        return np.median(pixels, axis=0) if pixels.size else None

    @staticmethod
    def _looks_like_background(frame, x, y, r, inner=None):
        """Árnyék vagy mintás padló: a kör belseje ugyanolyan színű, mint a közvetlen környezete,
        legfeljebb kicsit sötétebb / világosabb. Egy golyó vagy más színű, vagy jóval sötétebb / világosabb."""
        if inner is None:
            inner = CircleDetector._median_color(frame, x, y, 0, max(1.0, r * 0.6))
        ring = CircleDetector._median_color(frame, x, y, r * 1.3, r * 1.8)
        if inner is None or ring is None:
            return False
        low, high = BACKGROUND_BRIGHTNESS_RANGE
        brightness = inner.max() / max(ring.max(), 1.0)
        # Fényerőtől független színarány: árnyékban mindhárom csatorna nagyjából ugyanannyival sötétedik
        color_diff = np.abs(inner / max(inner.sum(), 1.0) - ring / max(ring.sum(), 1.0)).max()
        return low <= brightness <= high and color_diff <= BACKGROUND_MAX_COLOR_DIFF

    @staticmethod
    def _color_of(frame, circle):
        x, y, r = circle
        # Csak a belső részt nézzük, hogy a széle és a háttér ne számítson bele
        inner = CircleDetector._median_color(frame, x, y, 0, max(1.0, r * 0.6))
        if inner is None or CircleDetector._looks_like_background(frame, x, y, r, inner):
            return None
        return classify_color(inner)

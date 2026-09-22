

# Kamera: index (pl. 0) vagy videófájl elérési útja
CAMERA_SOURCE = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 60  # kért érték, a kamera ennél kevesebbet is adhat


# Kör-felismerés (cv2.HoughCircles, HOUGH_GRADIENT_ALT)
# A keresés ekkora szélességű, kicsinyített képen fut: kisebb = gyorsabb, de a kis körök elveszhetnek
PROCESS_WIDTH = 320
BLUR_KERNEL = 5
HOUGH_DP = 1.5
HOUGH_PARAM1 = 300   # Canny élkeresés felső küszöbe
HOUGH_PARAM2 = 0.8   # "kerekség" 0..1, nagyobb = szigorúbb, kevesebb hamis találat
# Az alábbiak az eredeti képkocka pixeleiben értendők
MIN_RADIUS = 10
MAX_RADIUS = 240
MIN_DIST = 20        # két kör középpontja közötti minimális távolság

# Színszűrés: csak a fekete / közel fekete körök maradnak.
# A kör belsejének medián fényessége (HSV V csatorna, 0..255) legfeljebb ennyi lehet; 255 = kikapcsolva
MAX_BRIGHTNESS = 70


# Követés
MAX_MATCH_DISTANCE = 80   # px, ennél messzebbi találatot nem párosítunk egy meglévő objektumhoz
MAX_MISSED_FRAMES = 8     # ennyi képkockán át becsüljük a helyét, ha nem látjuk
MIN_HITS = 3              # ennyi találat után jelenik meg (hamis találatok szűrése)
TRAIL_LENGTH = 30
PROCESS_NOISE = 1e-2      # Kalman: nagyobb = gyorsabban reagál a hirtelen irányváltásra
MEASUREMENT_NOISE = 1e-1  # Kalman: nagyobb = simább, de késleltetettebb pálya


SAVE_VIDEO = True
OUTPUT_VIDEO_PATH = "output/detections.avi"


ENABLE_SOUND_ALERT = True  # új kör alakú objektum megjelenésekor
ALERT_SOUND_PATH = "sounds/alert.wav"
ALERT_COOLDOWN = 5



# Kamera: index (pl. 0) vagy videófájl elérési útja, pl. egy "r" gombbal rögzített "output/raw_....avi"
CAMERA_SOURCE = 0
# A kért felbontás; ha a kamera nagyobbat ad (az Iriun pl. 1920x1080-at), CAMERA_WIDTH szélességre kicsinyítjük,
# hogy az alábbi pixelben megadott beállítások kamerától függetlenül ugyanazt jelentsék
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
DUPLICATE_OVERLAP = 0.8  # ha egy kör középpontja egy másik kör sugarának ennyiszeresén belül van, az ugyanaz a golyó

# Színfelismerés a kör belsejének medián színéből (HSV, 0..255)
BLACK_MAX_VALUE = 70       # ennél nem fényesebb = fekete
GRAY_MAX_SATURATION = 60   # ennél nem telítettebb = fehér / szürke
WHITE_MIN_VALUE = 180      # telítetlen és legalább ilyen fényes = fehér
# Csak ezek a színek maradnak, pl. {"fekete", "piros"}; None = minden szín
TRACKED_COLORS = None
TARGET_COLOR = "fekete"    # ez a cél golyó, X-szel jelölve
# Árnyék / padló kiszűrése: ha a kör belseje a környezetéhez képest ekkora fényerő-arányú (belső / külső)
# és ennél kisebb a színarány-eltérése, az nem golyó. Mért értékek: árnyék 0.92 / 0.014, padló 0.92-1.02 / 0.005,
# zöld labda 1.36 / 0.127, fehér labda 1.34 / 0.007
BACKGROUND_BRIGHTNESS_RANGE = (0.5, 1.15)
BACKGROUND_MAX_COLOR_DIFF = 0.03


# Követés
MAX_MATCH_DISTANCE = 80   # px, ennél messzebbi találatot nem párosítunk egy meglévő objektumhoz
MAX_MISSED_FRAMES = 8     # ennyi képkockán át becsüljük a helyét, ha nem látjuk
MIN_HITS = 3              # ennyi találat után jelenik meg (hamis találatok szűrése)
TRAIL_LENGTH = 30
ACCEL_NOISE = 1500        # px/s², Kalman: nagyobb = gyorsabban reagál a hirtelen irányváltásra, de zajosabb
MEASUREMENT_NOISE = 2.0   # px, a mért középpont szórása; nagyobb = simább, de késleltetettebb pálya
COLOR_HISTORY = 15        # a golyó színe az utolsó ennyi mérés többségi szavazata
COLOR_MISMATCH_PENALTY = 40  # px, eltérő színű találat párosításakor ennyivel "messzebbinek" számít
# Ha a körkeresés elveszít egy golyót (elmosódás, takarás), a várható helye körül a színe alapján keressük
COLOR_REDETECT = True
REDETECT_MIN_FILL = 0.4   # a talált színes folt területe a golyó körének legalább ennyiszerese legyen
REDETECT_MAX_FILL = 3.0   # ... és legfeljebb ennyiszerese (elmosódva nagyobb lehet)

# Pályajóslás
# A sebesség az utolsó VELOCITY_WINDOW mp mért pozícióira illesztett egyenesből jön; ha a pontok ennél
# jobban eltérnek az egyenestől (a golyó irányt vált, kézben van), nem jósolunk
VELOCITY_WINDOW = 0.3            # s
MIN_VELOCITY_POINTS = 5
MAX_FIT_ERROR = 3.0              # px
PREDICTION_SECONDS = 1.5         # legfeljebb ennyi időre előre jósolunk
PREDICTION_DECELERATION = 300    # px/s², a golyó lassulása gurulás közben (0 = nem lassul); a fapadlós felvételből mérve
MIN_PREDICTION_SPEED = 100       # px/s, ennél lassabb golyónál nem rajzolunk pályát
BOUNCE_ON_EDGES = False          # True: a jósolt pálya visszapattan a kép széléről (csak ha az fal is)


SAVE_VIDEO = True
OUTPUT_VIDEO_PATH = "output/detections.avi"
# Az "r" gombbal indított nyers felvétel (jelölések nélkül) ide kerül; CAMERA_SOURCE-ként visszajátszható
RAW_VIDEO_DIR = "output"


ENABLE_SOUND_ALERT = True  # új kör alakú objektum megjelenésekor
ALERT_SOUND_PATH = "sounds/alert.wav"
ALERT_COOLDOWN = 5

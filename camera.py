
import threading
import time
import cv2


class CameraStream:
    """Élő kameránál külön szálon olvas, így a feldolgozás mindig a legfrissebb képkockát kapja,
    és nem gyűlik fel késleltetés a kamera pufferében. Videófájlt kockáról kockára, kihagyás nélkül ad vissza,
    hogy egy felvétel visszajátszása mindig ugyanazt az eredményt adja.
    Minden képkockához időbélyeg (mp) tartozik: élő kameránál a beérkezés ideje, fájlnál a videóbeli idő."""

    def __init__(self, source, width, height, fps):
        self.cap = cv2.VideoCapture(source)
        self.width = width
        self.is_file = isinstance(source, str)
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame = None
        self.frame_id = 0
        self.timestamp = 0.0
        self.running = False
        self._cond = threading.Condition()
        self._thread = None

    @property
    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def start(self):
        if not self._grab():
            return False
        self.running = True
        if not self.is_file:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return True

    def _grab(self):
        ok, frame = self.cap.read()
        if not ok:
            return False
        frame = self._fit(frame)
        if self.is_file:
            timestamp = self.frame_id / (self.fps or 30.0)
        else:
            timestamp = time.perf_counter()
        self.frame, self.timestamp = frame, timestamp
        self.frame_id += 1
        return True

    def _run(self):
        while self.running:
            ok, frame = self.cap.read()
            now = time.perf_counter()
            with self._cond:
                if not ok:
                    self.running = False
                else:
                    self.frame, self.timestamp = self._fit(frame), now
                    self.frame_id += 1
                self._cond.notify_all()

    def _fit(self, frame):
        # Nagyobb képet (pl. Iriun 1920x1080) a kért szélességre kicsinyítünk
        if frame.shape[1] > self.width:
            frame = cv2.resize(frame, None, fx=self.width / frame.shape[1], fy=self.width / frame.shape[1],
                               interpolation=cv2.INTER_AREA)
        return frame

    def read(self, last_id, timeout=1.0):
        """Megvárja a last_id-nél újabb képkockát.
        Visszatérés: (frame, frame_id, timestamp), vagy (None, last_id, None) ha nincs új."""
        if self.is_file:
            if not self._grab():
                return None, last_id, None
            return self.frame, self.frame_id, self.timestamp

        with self._cond:
            self._cond.wait_for(lambda: self.frame_id != last_id or not self.running, timeout)
            if self.frame_id == last_id:
                return None, last_id, None
            return self.frame, self.frame_id, self.timestamp

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.cap.release()

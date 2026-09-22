
import threading
import cv2


class CameraStream:
    """Külön szálon olvassa a kamerát, így a feldolgozás mindig a legfrissebb képkockát kapja,
    és nem gyűlik fel késleltetés a kamera pufferében."""

    def __init__(self, source, width, height, fps):
        self.cap = cv2.VideoCapture(source)
        self.is_file = isinstance(source, str)
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame = None
        self.frame_id = 0
        self.running = False
        self._cond = threading.Condition()
        self._thread = None

    @property
    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def start(self):
        ok, frame = self.cap.read()
        if not ok:
            return False
        self.frame, self.frame_id = frame, 1
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def _run(self):
        while self.running:
            ok, frame = self.cap.read()
            with self._cond:
                if not ok:
                    self.running = False
                else:
                    self.frame = frame
                    self.frame_id += 1
                self._cond.notify_all()

    def read(self, last_id, timeout=1.0):
        """Megvárja a last_id-nél újabb képkockát. Visszatérés: (frame, frame_id), vagy (None, last_id) ha nincs új."""
        with self._cond:
            self._cond.wait_for(lambda: self.frame_id != last_id or not self.running, timeout)
            if self.frame_id == last_id:
                return None, last_id
            return self.frame, self.frame_id

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.cap.release()

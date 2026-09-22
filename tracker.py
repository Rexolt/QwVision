
from collections import deque
import cv2
import numpy as np
from config import MAX_MATCH_DISTANCE, MAX_MISSED_FRAMES, MIN_HITS, TRAIL_LENGTH, PROCESS_NOISE, MEASUREMENT_NOISE


class Track:
    def __init__(self, track_id, x, y, r):
        self.id = track_id
        self.radius = float(r)
        self.hits = 1
        self.missed = 0
        self.trail = deque([(int(x), int(y))], maxlen=TRAIL_LENGTH)

        # Állandó sebességű modell: állapot (x, y, vx, vy), mérés (x, y)
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * PROCESS_NOISE
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * MEASUREMENT_NOISE
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
        self.kf = kf

    @property
    def position(self):
        return float(self.kf.statePost[0, 0]), float(self.kf.statePost[1, 0])

    @property
    def velocity(self):
        return float(self.kf.statePost[2, 0]), float(self.kf.statePost[3, 0])

    @property
    def confirmed(self):
        return self.hits >= MIN_HITS

    def predict(self):
        self.kf.predict()
        return self.position

    def update(self, x, y, r):
        self.kf.correct(np.array([[x], [y]], np.float32))
        self.radius = 0.6 * self.radius + 0.4 * float(r)
        self.hits += 1
        self.missed = 0
        self._add_trail_point()

    def mark_missed(self):
        self.missed += 1
        self._add_trail_point()

    def _add_trail_point(self):
        x, y = self.position
        self.trail.append((int(x), int(y)))


class CircleTracker:
    def __init__(self):
        self.tracks = []
        self._next_id = 1

    def update(self, detections):
        """Detekciók (N, 3) hozzárendelése a követett objektumokhoz. Visszatérés: az ebben a lépésben megerősített követések."""
        predicted = np.array([t.predict() for t in self.tracks], np.float32).reshape(-1, 2)

        matches = {}  # követés indexe -> detekció indexe
        if len(self.tracks) and len(detections):
            dists = np.linalg.norm(predicted[:, None, :] - detections[None, :, :2], axis=2)
            # Mohó párosítás a legközelebbi pároktól kezdve
            used_dets = set()
            for flat in np.argsort(dists, axis=None):
                ti, di = divmod(int(flat), dists.shape[1])
                if dists[ti, di] > MAX_MATCH_DISTANCE:
                    break
                if ti in matches or di in used_dets:
                    continue
                matches[ti] = di
                used_dets.add(di)

        newly_confirmed = []
        for ti, track in enumerate(self.tracks):
            if ti in matches:
                was_confirmed = track.confirmed
                track.update(*detections[matches[ti]])
                if track.confirmed and not was_confirmed:
                    newly_confirmed.append(track)
            else:
                track.mark_missed()

        self.tracks = [t for t in self.tracks if t.missed <= MAX_MISSED_FRAMES]

        matched_dets = set(matches.values())
        for di, (x, y, r) in enumerate(detections):
            if di not in matched_dets:
                self.tracks.append(Track(self._next_id, x, y, r))
                self._next_id += 1

        return newly_confirmed

    def visible_tracks(self):
        return [t for t in self.tracks if t.confirmed]

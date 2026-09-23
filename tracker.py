
from collections import Counter, deque
import math
import cv2
import numpy as np
from config import (MAX_MATCH_DISTANCE, MAX_MISSED_FRAMES, MIN_HITS, TRAIL_LENGTH, ACCEL_NOISE, MEASUREMENT_NOISE,
                    COLOR_HISTORY, COLOR_MISMATCH_PENALTY, PREDICTION_SECONDS, PREDICTION_DECELERATION,
                    BOUNCE_ON_EDGES, VELOCITY_WINDOW, MIN_VELOCITY_POINTS, MAX_FIT_ERROR, MIN_PREDICTION_SPEED)

PREDICTION_STEP = 1 / 60  # s, a jósolt pálya pontjainak sűrűsége


class Track:
    def __init__(self, track_id, x, y, r, color, timestamp):
        self.id = track_id
        self.radius = float(r)
        self.color_votes = deque([color], maxlen=COLOR_HISTORY)
        self.color = color
        self.hits = 1
        self.missed = 0
        self.trail = deque([(int(x), int(y))], maxlen=TRAIL_LENGTH)
        self.measurements = deque([(timestamp, float(x), float(y))], maxlen=64)  # (t, x, y), csak a mért helyek

        # Állandó sebességű modell valódi idővel: állapot (x, y, vx, vy) px és px/s, mérés (x, y)
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * MEASUREMENT_NOISE ** 2
        # A kezdeti sebességet nem ismerjük, ezért annak nagy a bizonytalansága
        kf.errorCovPost = np.diag([MEASUREMENT_NOISE ** 2] * 2 + [1000.0 ** 2] * 2).astype(np.float32)
        kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
        self.kf = kf

    @property
    def position(self):
        return float(self.kf.statePost[0, 0]), float(self.kf.statePost[1, 0])

    @property
    def velocity(self):
        """px/s"""
        return float(self.kf.statePost[2, 0]), float(self.kf.statePost[3, 0])

    @property
    def confirmed(self):
        return self.hits >= MIN_HITS

    def predict(self, dt):
        # Az Iriun képkockái nem egyenletesen érkeznek, ezért a modell a valódi eltelt idővel számol
        self.kf.transitionMatrix = np.array([[1, 0, dt, 0],
                                             [0, 1, 0, dt],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        # Véletlen gyorsulás (ütközés, lökés) hatása dt alatt
        q = ACCEL_NOISE ** 2
        pp, pv, vv = q * dt ** 4 / 4, q * dt ** 3 / 2, q * dt ** 2
        self.kf.processNoiseCov = np.array([[pp, 0, pv, 0],
                                            [0, pp, 0, pv],
                                            [pv, 0, vv, 0],
                                            [0, pv, 0, vv]], np.float32)
        self.kf.predict()
        return self.position

    def update(self, x, y, r, color, timestamp):
        self.kf.correct(np.array([[x], [y]], np.float32))
        self.measurements.append((timestamp, float(x), float(y)))
        self.radius = 0.6 * self.radius + 0.4 * float(r)
        # Többségi szavazás, hogy egy-egy rosszul megvilágított képkocka ne váltsa át a színt
        self.color_votes.append(color)
        self.color = Counter(self.color_votes).most_common(1)[0][0]
        self.hits += 1
        self.missed = 0
        self._add_trail_point()

    def mark_missed(self):
        self.missed += 1
        self._add_trail_point()

    def _add_trail_point(self):
        x, y = self.position
        self.trail.append((int(x), int(y)))

    def fitted_motion(self):
        """Egyenes illesztése az utolsó VELOCITY_WINDOW mp mért pozícióira.
        Visszatérés: ((x, y) az utolsó mérés idején, (vx, vy) px/s), vagy None, ha nincs elég pont,
        vagy a golyó nem egyenesen halad (irányt váltott, kézben van, zajos a mérés)."""
        last_t = self.measurements[-1][0]
        points = np.array([m for m in self.measurements if m[0] >= last_t - VELOCITY_WINDOW])
        if len(points) < MIN_VELOCITY_POINTS:
            return None
        t = points[:, 0] - last_t
        vx, x0 = np.polyfit(t, points[:, 1], 1)
        vy, y0 = np.polyfit(t, points[:, 2], 1)
        error = np.sqrt(np.mean((points[:, 1] - (vx * t + x0)) ** 2 + (points[:, 2] - (vy * t + y0)) ** 2))
        if error > MAX_FIT_ERROR:
            return None
        return (x0, y0), (vx, vy)

    def predicted_path(self, width, height):
        """Várható útvonal pontjai: a golyó egyenletesen lassul (gördülési ellenállás).
        None, ha nem megbízható: nem látjuk épp, a kép szélébe lóg (a középpontja ott pontatlan),
        túl lassú, vagy nem egyenesen halad."""
        x, y = self.position
        r = self.radius
        if self.missed or not (r <= x <= width - r and r <= y <= height - r):
            return None
        motion = self.fitted_motion()
        if motion is None or math.hypot(*motion[1]) < MIN_PREDICTION_SPEED:
            return None
        (x, y), (vx, vy) = motion
        r = min(r, width / 2, height / 2)
        path = [(x, y)]
        for _ in range(int(PREDICTION_SECONDS / PREDICTION_STEP)):
            speed = math.hypot(vx, vy)
            new_speed = speed - PREDICTION_DECELERATION * PREDICTION_STEP
            if new_speed <= 0:
                break
            vx, vy = vx * new_speed / speed, vy * new_speed / speed
            x += vx * PREDICTION_STEP
            y += vy * PREDICTION_STEP
            if BOUNCE_ON_EDGES:
                if x < r or x > width - r:
                    x = 2 * (r if x < r else width - r) - x
                    vx = -vx
                if y < r or y > height - r:
                    y = 2 * (r if y < r else height - r) - y
                    vy = -vy
            elif not (0 <= x <= width and 0 <= y <= height):
                break
            path.append((x, y))
        return path


class CircleTracker:
    def __init__(self):
        self.tracks = []
        self._next_id = 1
        self._last_time = None

    def update(self, detections, colors, timestamp, redetect=None):
        """Detekciók (N, 3) és színeik hozzárendelése a követett objektumokhoz.
        redetect(x, y, r, color) -> (x, y, r) | None: tartalék keresés azokra a golyókra, amelyeket a körkeresés
        ebben a képkockában nem talált meg.
        Visszatérés: az ebben a lépésben megerősített követések."""
        dt = 1 / 30 if self._last_time is None else min(max(timestamp - self._last_time, 1e-3), 0.5)
        self._last_time = timestamp
        predicted = np.array([t.predict(dt) for t in self.tracks], np.float32).reshape(-1, 2)

        matches = {}  # követés indexe -> detekció indexe
        if len(self.tracks) and len(detections):
            dists = np.linalg.norm(predicted[:, None, :] - detections[None, :, :2], axis=2)
            # Eltérő színű golyót csak akkor párosítunk, ha nincs jobb jelölt (pl. két összeérő golyó)
            mismatch = np.array([[t.color != c for c in colors] for t in self.tracks])
            dists += mismatch * COLOR_MISMATCH_PENALTY
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
                di = matches[ti]
                track.update(*detections[di], colors[di], timestamp)
                if track.confirmed and not was_confirmed:
                    newly_confirmed.append(track)

        matched_dets = set(matches.values())
        for ti, track in enumerate(self.tracks):
            if ti in matches:
                continue
            found = redetect(*track.position, track.radius, track.color) if redetect and track.confirmed else None
            # Egy másik golyóhoz már hozzárendelt kört nem veszünk el
            if found is not None and not any(
                    math.hypot(found[0] - x, found[1] - y) < r for x, y, r in detections[list(matched_dets)]):
                track.update(*found, track.color, timestamp)
            else:
                track.mark_missed()

        self.tracks = [t for t in self.tracks if t.missed <= MAX_MISSED_FRAMES]

        for di, (x, y, r) in enumerate(detections):
            if di not in matched_dets:
                self.tracks.append(Track(self._next_id, x, y, r, colors[di], timestamp))
                self._next_id += 1

        return newly_confirmed

    def visible_tracks(self):
        return [t for t in self.tracks if t.confirmed]

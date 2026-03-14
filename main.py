import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Thread
from urllib.request import urlretrieve

import cv2
import mediapipe as mp


MODEL_DIR = Path("models/mediapipe")
MODEL_PATH = MODEL_DIR / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


@dataclass
class Config:
    camera_id: int = 0
    width: int = 640
    height: int = 480
    num_poses: int = 1
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    visibility_threshold: float = 0.45
    smoothing: float = 0.8
    hold_frames: int = 8


@dataclass
class LandmarkState:
    x: float | None = None
    y: float | None = None
    visibility: float = 0.0
    missing_frames: int = 0


class FastCamera:
    def __init__(self, cfg: Config):
        self.cap = cv2.VideoCapture(cfg.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Kamera acilamadi: {cfg.camera_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = Lock()
        self.frame = None
        self.running = True
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


class StablePoseRenderer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.states = [LandmarkState() for _ in range(33)]
        connection_source = mp.tasks.vision.PoseLandmarksConnections
        self.connections = tuple(
            getattr(connection_source, "POSE_LANDMARKS", ())
        )

    def update(self, landmarks):
        if not landmarks:
            self._handle_missing()
            return

        pose = landmarks[0]
        for index, landmark in enumerate(pose):
            state = self.states[index]
            visible = landmark.visibility >= self.cfg.visibility_threshold

            if not visible:
                if state.x is not None and state.missing_frames < self.cfg.hold_frames:
                    state.missing_frames += 1
                else:
                    state.x = None
                    state.y = None
                    state.visibility = 0.0
                    state.missing_frames = 0
                continue

            if state.x is None or state.y is None:
                state.x = landmark.x
                state.y = landmark.y
            else:
                alpha = self._adaptive_smoothing(state.x, state.y, landmark.x, landmark.y)
                state.x = state.x * alpha + landmark.x * (1.0 - alpha)
                state.y = state.y * alpha + landmark.y * (1.0 - alpha)

            state.visibility = landmark.visibility
            state.missing_frames = 0

    def _adaptive_smoothing(self, old_x, old_y, new_x, new_y):
        movement = max(abs(new_x - old_x), abs(new_y - old_y))
        if movement > 0.08:
            return max(0.55, self.cfg.smoothing - 0.22)
        if movement > 0.04:
            return max(0.65, self.cfg.smoothing - 0.12)
        return self.cfg.smoothing

    def _handle_missing(self):
        for state in self.states:
            if state.x is None:
                continue
            if state.missing_frames < self.cfg.hold_frames:
                state.missing_frames += 1
            else:
                state.x = None
                state.y = None
                state.visibility = 0.0
                state.missing_frames = 0

    def draw(self, frame):
        height, width = frame.shape[:2]

        for connection in self.connections:
            start = self.states[connection.start]
            end = self.states[connection.end]
            if start.x is None or end.x is None:
                continue

            p1 = (int(start.x * width), int(start.y * height))
            p2 = (int(end.x * width), int(end.y * height))
            cv2.line(frame, p1, p2, (0, 255, 180), 3, cv2.LINE_AA)

        for state in self.states:
            if state.x is None:
                continue
            point = (int(state.x * width), int(state.y * height))
            cv2.circle(frame, point, 4, (255, 255, 255), -1, cv2.LINE_AA)


def ensure_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        return
    print("Pose Landmarker modeli indiriliyor...")
    urlretrieve(MODEL_URL, MODEL_PATH)


def main():
    cfg = Config()
    ensure_model()

    base_options = mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=cfg.num_poses,
        min_pose_detection_confidence=cfg.min_pose_detection_confidence,
        min_pose_presence_confidence=cfg.min_pose_presence_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
        output_segmentation_masks=False,
    )

    camera = FastCamera(cfg)
    renderer = StablePoseRenderer(cfg)
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    previous_time = time.perf_counter()
    fps = 0.0
    print("Sistem calisiyor. Cikmak icin Q tusuna basin.")

    try:
        while True:
            frame = camera.read()
            if frame is None:
                time.sleep(0.005)
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            renderer.update(result.pose_landmarks)
            renderer.draw(frame)

            now = time.perf_counter()
            delta = now - previous_time
            previous_time = now
            if delta > 0:
                instant_fps = 1.0 / delta
                fps = instant_fps if fps == 0.0 else fps * 0.85 + instant_fps * 0.15

            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Cikis: Q",
                (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Stable Pose Tracking", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        camera.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

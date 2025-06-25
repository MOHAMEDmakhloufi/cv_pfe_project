from dataclasses import dataclass, field
import math

@dataclass
class TrackedObject:
    track_id: int
    class_id: int
    class_name: str
    conf: float
    box: any  # YOLO Box object
    is_mobile: bool
    track_history: list = field(default_factory=list) # List of (x, y, frame_nb)
    _speed: float = 0.0
    _angle_deg: float = None
    _vector: tuple = field(default_factory=lambda: (0.0, 0.0))
    _depth: float = None
    _best_frame_for_depth: int = None

    def get_track_id(self):
        return self.track_id

    def get_class_id(self):
        return self.class_id

    def get_class_name(self):
        return self.class_name

    def get_conf(self):
        return self.conf

    def get_box(self):
        return self.box

    def get_is_mobile(self):
        return self.is_mobile

    def get_track_history(self):
        return self.track_history

    def get_speed(self):
        return self._speed

    def get_angle_deg(self):
        return self._angle_deg

    def get_vector(self):
        return self._vector

    def get_depth(self):
        return self._depth

    def get_best_frame_for_depth(self):
        return self._best_frame_for_depth

    def set_depth(self, depth):
        self._depth = depth

    def set_best_frame_for_depth(self, frame_id):
        self._best_frame_for_depth = frame_id

    def add_to_history(self, center_point):
        self.track_history.append(center_point)

    def compute_average_motion(self, sequence_len_s):
        if len(self.track_history) < 2 or not self.is_mobile:
            return {"speed": 0.0, "angle_deg": None, "vector": (0.0, 0.0)}

        total_distance = 0.0
        total_dx = 0.0
        total_dy = 0.0
        steps = len(self.track_history) - 1

        for i in range(steps):
            x1, y1, _ = self.track_history[i]
            x2, y2, _ = self.track_history[i + 1]

            dx = x2 - x1
            dy = y2 - y1
            total_dx += dx
            total_dy += dy

            dist = math.sqrt(dx ** 2 + dy ** 2)
            total_distance += dist

        avg_distance_per_frame = total_distance / steps
        speed = avg_distance_per_frame / sequence_len_s

        avg_dx = total_dx / steps
        avg_dy = total_dy / steps

        angle_rad = math.atan2(avg_dy, avg_dx)
        angle_deg = math.degrees(angle_rad) % 360

        return {
            "speed": speed,
            "angle_deg": angle_deg,
            "vector": (avg_dx, avg_dy)
        }

    def update_motion_attributes(self, motion_data):
        self._speed = motion_data["speed"]
        self._angle_deg = motion_data["angle_deg"]
        self._vector = motion_data["vector"]

    def __str__(self):
        return (f"TrackedObject(trackId={self.track_id}, classId={self.class_id}, "
                f"className=\'{self.class_name}\' conf={self.conf:.2f}, isMobile={self.is_mobile}, )")



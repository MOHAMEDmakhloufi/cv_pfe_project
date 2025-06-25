class SceneFilter:
    def __init__(self):
        pass

    def filter_mobile_objects_by_direction(self, tracked_objects, frame_shape):
        filtered_objects = dict()

        for key, obj in tracked_objects.items():
            if not obj.get_is_mobile():
                filtered_objects[key] = obj
                continue

            history = obj.get_track_history()
            if not history or len(history) < 2:
                continue

            last_x, last_y, f = history[-1]
            angle = obj.get_angle_deg()

            frame_center_x = frame_shape[0] / 2
            frame_center_y = frame_shape[1] / 2

            if (
                (last_x > frame_center_x and 135 <= angle <= 225) or  # right side, moving left
                (last_x < frame_center_x and (angle <= 45 or angle >= 315)) or  # left side, moving right
                (last_y > frame_center_y and 45 < angle < 135) or  # bottom, moving up
                (last_y < frame_center_y and 225 < angle < 315)    # top, moving down
            ):
                filtered_objects[key] = obj

        return filtered_objects

    def _filter_objects_by_depth(self, tracked_objects, frames_depth_estimation, depth_threshold, is_mobile_filter=None, edge_margin=None, frame_shape=None):
        filtered_objects = dict()

        x_left_bound = 0
        x_right_bound = float('inf')
        if edge_margin is not None and frame_shape is not None:
            frame_width = frame_shape[1]
            x_left_bound = frame_width * edge_margin
            x_right_bound = frame_width * (1 - edge_margin)

        for key, obj in tracked_objects.items():
            if is_mobile_filter is not None and obj.get_is_mobile() != is_mobile_filter:
                filtered_objects[key] = obj
                continue

            track_history = obj.get_track_history()

            for x, y, frame_id in reversed(track_history):
                if edge_margin is not None and not (x_left_bound <= x <= x_right_bound):
                    continue

                if frame_id in frames_depth_estimation:
                    depth_map = frames_depth_estimation[frame_id]

                    depth_value = float(depth_map[int(y), int(x)])
                    obj.set_depth(depth_value)

                    if depth_value >= depth_threshold:
                        filtered_objects[key] = obj
                    break
        return filtered_objects

    def filter_static_objects_by_depth(self, tracked_objects, frames_depth_estimation, frame_shape, edge_margin=0.1, depth_threshold=0.5):
        return self._filter_objects_by_depth(tracked_objects, frames_depth_estimation, depth_threshold, is_mobile_filter=False, edge_margin=edge_margin, frame_shape=frame_shape)

    def filter_mobile_objects_by_depth(self, tracked_objects, frames_depth_estimation, depth_threshold=0.5):
        return self._filter_objects_by_depth(tracked_objects, frames_depth_estimation, depth_threshold, is_mobile_filter=True)



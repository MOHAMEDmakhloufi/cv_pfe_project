from src.data_models.tracked_object import TrackedObject

class ObjectTracker:
    def __init__(self, classes_names, mobile_classes_names):
        self.classes_names = classes_names
        self.mobile_classes_names = mobile_classes_names

    def parse_sequence_tracking_results(self, tracked_objects, sequence_len_s):
        for obj in tracked_objects.values():
            if obj.get_is_mobile():
                motion_data = obj.compute_average_motion(sequence_len_s)
                obj.update_motion_attributes(motion_data)
        return tracked_objects

    def parse_tracking_result(self, result: any, tracked_objects: dict, frame_nb: int):
        if result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes[i]
                x_min, y_min, x_max, y_max = box.xyxy.cpu().numpy().squeeze().tolist()
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2

                track_id = int(box.id.item()) if box.id is not None else None
                class_id = int(box.cls.item())
                class_name = self.classes_names[class_id]
                conf = float(box.conf.item())

                if track_id not in tracked_objects:
                    obj = TrackedObject(
                        track_id=track_id,
                        class_id=class_id,
                        class_name=class_name,
                        conf=conf,
                        box=box,
                        is_mobile=class_name in self.mobile_classes_names,
                        track_history=[(float(x_center), float(y_center), frame_nb)]
                    )
                    tracked_objects[track_id] = obj
                else:
                    obj = tracked_objects[track_id]
                    obj.box = box
                    obj.conf = max(obj.get_conf(), conf)
                    obj.add_to_history((float(x_center), float(y_center), frame_nb))
        return tracked_objects



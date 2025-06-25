class SceneDescriber:
    def __init__(self):
        pass

    def generate_scene_description(self, tracked_objects, road_judging_classes):
        sorted_objects = sorted(tracked_objects.values(), key=lambda obj: obj.get_depth() or 1.0)

        categories = {
            "mobile": [],
            "static": [],
            "road_judging": []
        }

        for obj in sorted_objects:
            class_name = obj.get_class_name()

            if obj.get_is_mobile():
                categories["mobile"].append((class_name, obj.get_depth() or 1.0))
            elif class_name in road_judging_classes:
                categories["road_judging"].append((class_name, obj.get_depth() or 1.0))
            else:
                categories["static"].append((class_name, obj.get_depth() or 1.0))

        description = []

        if categories["road_judging"]:
            unique_road_objects = list(dict.fromkeys([obj[0] for obj in categories["road_judging"]]))
            if len(unique_road_objects) == 1:
                description.append(f"a {unique_road_objects[0]} ahead")
            else:
                description.append(f"{', '.join(unique_road_objects)} ahead")

        if categories["mobile"]:
            count = len(categories["mobile"])
            closest_mobile = min(categories["mobile"], key=lambda x: x[1])[0]
            if count == 1:
                description.append(f"a {closest_mobile} nearby")
            else:
                description.append(f"{count} moving objects, including a {closest_mobile}")

        if categories["static"]:
            close_static = [obj[0] for obj in categories["static"] if obj[1] < 2.0]
            if close_static:
                if len(close_static) == 1:
                    description.append(f"a {close_static[0]} obstacle")
                else:
                    description.append(f"obstacles, including a {close_static[0]}")

        if not description:
            return None

        if len(description) == 1:
            return f"Watch for {description[0]}."
        elif len(description) == 2:
            return f"Watch for {description[0]} and {description[1]}."
        else:
            return f"Watch for {', '.join(description[:-1])}, and {description[-1]}."

    def generate_scene_description_for_unknown(self, exist_unknown_object):
        if exist_unknown_object:
            return "Watch for unknown objects."
        return None



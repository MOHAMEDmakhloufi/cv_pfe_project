from ultralytics import YOLO
from src.modules.object_detection.detector import Detector

class YoloDetector(Detector):
    def __init__(self, model_path, conf_threshold, device):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device

    def detect(self, image):
        results = self.model(image, device=self.device)
        if results:
            return results[0]
        return None

    def object_tracking(self, image):
        results = self.model.track(
            image,
            show=False,
            save=False,
            conf=self.conf_threshold,
            persist=True,
            device=self.device,
            verbose=False
        )
        if results:
            return results[0]
        return None



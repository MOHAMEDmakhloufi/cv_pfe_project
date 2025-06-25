from abc import ABC, abstractmethod

class Detector(ABC):
    @abstractmethod
    def detect(self, image):
        pass

    @abstractmethod
    def object_tracking(self, image):
        pass



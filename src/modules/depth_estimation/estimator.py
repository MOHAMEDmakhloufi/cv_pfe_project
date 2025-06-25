from abc import ABC, abstractmethod
import numpy as np

class Estimator(ABC):
    @abstractmethod
    def estimate_depth(self, image_np) -> np.ndarray:
        pass

    @abstractmethod
    def estimate_depth_from_frames(self, frames_dict) -> dict:
        pass

    @abstractmethod
    def estimate_depth_from_optimal_frames(self, frames_list, tracked_objects) -> dict:
        pass



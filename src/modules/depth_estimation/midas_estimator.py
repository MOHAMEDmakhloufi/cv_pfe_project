import torch
import numpy as np
import warnings
import cv2
from src.modules.depth_estimation.estimator import Estimator

warnings.filterwarnings("ignore", category=FutureWarning)

class MidasEstimator(Estimator):
    def __init__(self, model_type="DPT_Hybrid", device="cpu", activate_logs=False):
        self.device = self._select_device(device, activate_logs)
        self.model, self.transform = self._load_midas_model(model_type, self.device, activate_logs)

    def _select_device(self, preferred_device="auto", activate_logs=True):
        if preferred_device.lower() == "cuda" and torch.cuda.is_available():
            if activate_logs: print("Using CUDA device.")
            return torch.device("cuda")
        elif preferred_device.lower() == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if activate_logs: print("Using MPS device (Apple Silicon).")
            return torch.device("mps")
        else:
            if preferred_device.lower() not in ["auto", "cpu"] :
                if activate_logs: print(f"Preferred device \'{preferred_device}\' not available or not supported. Falling back to CPU.")
            else:
                if activate_logs: print("Using CPU device.")
            return torch.device("cpu")

    def _load_midas_model(self, model_type="MiDaS_small", device=torch.device("cpu"), activate_logs=True):
        try:
            if activate_logs:
                print(f"Loading MiDaS model: {model_type} from PyTorch Hub...")
            midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
            midas.to(device)
            midas.eval()

            if activate_logs:
                print(f"Loading MiDaS transforms for {model_type}...")
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            if model_type == "dpt_large" or model_type == "dpt_hybrid":
                transform = midas_transforms.dpt_transform
            else:
                transform = midas_transforms.small_transform

            if activate_logs:
                print("Model and transforms loaded successfully.")
            return midas, transform
        except Exception as e:
            print(f"Error loading MiDaS model or transforms: {e}")
            print("Please ensure you have an internet connection and necessary permissions.")
            return None, None

    def estimate_depth(self, image_np) -> np.ndarray:
        if image_np is None:
            return None
        try:
            input_batch = self.transform(image_np).to(self.device)
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image_np.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            depth_map = prediction.cpu().numpy()
            return depth_map
        except Exception as e:
            print(f"Error during depth prediction: {e}")
            return None

    def save_depth_map_heatmap(self,depth_map: np.ndarray, file_path: str):
        import matplotlib.pyplot as plt
        if depth_map is not None:
            # Create a new figure
            plt.figure()
            # Display depth map with 'inferno' colormap
            plt.imshow(depth_map, cmap='inferno')
            plt.title("Estimated Depth")
            plt.axis('off')
            # Save the figure
            plt.savefig(file_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()  # Close the figure to free memory
            print(f"Depth map saved as {file_path}")
        else:
            print("No depth map to save")

    def _normalize_depth_map(self, depth_map):
        depth_map = np.maximum(depth_map, 0)
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 1e-6:
            normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            normalized_depth = np.zeros_like(depth_map)
        return normalized_depth

    def _normalize_to_0_255(self, img):
        normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)

    def get_visual_depth_map(self, depth_map):
        depth_visual = self._normalize_to_0_255(depth_map)
        return cv2.cvtColor(depth_visual, cv2.COLOR_GRAY2BGR)

    def _get_optimal_frames(self, tracked_objects):
        tracked_objects_frames = dict()
        for key, obj in tracked_objects.items():
            seen_obj = set(map(lambda x: x[2], obj.get_track_history()))
            tracked_objects_frames[key] = seen_obj

        frame_to_objects = {}
        for obj_id, frames in tracked_objects_frames.items():
            for frame_id in frames:
                if frame_id not in frame_to_objects:
                    frame_to_objects[frame_id] = set()
                frame_to_objects[frame_id].add(obj_id)

        uncovered_objects = set(tracked_objects_frames.keys())
        selected_frames = []

        while uncovered_objects:
            max_coverage = 0
            best_frame = None
            for frame_id, objects in frame_to_objects.items():
                coverage = len(objects & uncovered_objects)
                if coverage > max_coverage:
                    max_coverage = coverage
                    best_frame = frame_id

            if best_frame is None:
                break

            selected_frames.append(best_frame)
            uncovered_objects -= frame_to_objects[best_frame]
            del frame_to_objects[best_frame]

        return sorted(selected_frames)

    def estimate_depth_from_frames(self, frames_dict) -> dict:
        frames_depth_estimation = dict()
        import os
        output_dir = "../outputs/frames_depth_estimation/"
        os.makedirs(output_dir, exist_ok=True)
        for frame_id, frame in frames_dict.items():
            sample_depth_map = self.estimate_depth(frame)
            #output_path = os.path.join(output_dir, f"depth_map_{frame_id}.jpg")
            #self.save_depth_map_heatmap(sample_depth_map, output_path)
            normalized_depth_map = self._normalize_depth_map(sample_depth_map)
            frames_depth_estimation[frame_id] = normalized_depth_map
        return frames_depth_estimation

    def estimate_depth_from_optimal_frames(self, frames_list, tracked_objects) -> dict:
        selected_frames = self._get_optimal_frames(tracked_objects)
        frames_dict = dict()
        for frame_id in selected_frames:
            frames_dict[frame_id] = frames_list[frame_id]
        return self.estimate_depth_from_frames(frames_dict)



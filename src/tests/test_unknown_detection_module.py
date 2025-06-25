import cv2
import numpy as np
from sympy.physics.quantum.gate import normalized

from src.core.config_manager import ConfigManager
from src.core.logger import setup_logging
from src.core.pipeline_manager import PipelineManager

from src.modules.object_detection.yolo_detector import YoloDetector
from src.modules.depth_estimation.midas_estimator import MidasEstimator
from src.modules.tracking.tracker import ObjectTracker
from src.modules.scene_analysis.filter import SceneFilter
from src.modules.unknown_detection.unknown_detector import UnknownDetector
from src.modules.scene_analysis.describer import SceneDescriber
from src.modules.output.gtts_synthesizer import GTTSSpeechSynthesizer

from src.utils import read_dataset_conf

def draw_boundaries(image, edge_margin):

    img_height, img_width = image.shape[:2]

    # Compute boundaries
    x_left_bound = int(img_width * edge_margin)
    x_right_bound = int(img_width * (1 - edge_margin))

    # Draw transparent red regions for edge bounds
    overlay = image.copy()
    alpha = 0.5  # transparency

    # Left margin
    cv2.rectangle(overlay, (0, 0), (x_left_bound, img_height), (0, 0, 255), -1)
    # Right margin
    cv2.rectangle(overlay, (x_right_bound, 0), (img_width, img_height), (0, 0, 255), -1)

    # Apply transparency blend
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

if __name__ == '__main__':
    logger = setup_logging()

    # Initialize ConfigManager
    config_manager = ConfigManager.initialize(
        main_config_path="../../configs/main_config.yaml",
        dataset_config_path="../../configs/dataset_config.yaml",
        models_config_path="../../configs/models_config.yaml"
    )

    # Load configurations
    main_config = config_manager.get_config("main")
    models_config = config_manager.get_config("models")

    depth_estimator = MidasEstimator(
        model_type=models_config["midas_model_type"],
        device=models_config["midas_device"]
    )

    unknown_detector = UnknownDetector()

    unknown_edge_margin = main_config.get("unknown_detection", {}).get("edge_margin", 0.1)
    unknown_box_h_threshold = main_config.get("unknown_detection", {}).get("box_h_threshold", 10)
    depth_threshold = models_config["depth_threshold_static_obj"]

    # Initialize and run the test
    source = main_config.get("video_source")
    cap = cv2.VideoCapture('../'+source)

    if not cap.isOpened():
        logger.error(f"Error: Could not open video source: {source}")
        exit()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream.")
                break

            # Simulated depth map (0=far, 1=near)
            sample_depth_map = depth_estimator.estimate_depth(frame)
            normalized_depth = depth_estimator._normalize_depth_map(sample_depth_map)
            normalized_depth[normalized_depth < depth_threshold] = 0.0

            # Detect unknowns
            exists, bbox, contour_img = unknown_detector.has_unknown_object(normalized_depth, unknown_edge_margin, unknown_box_h_threshold)
            x2, y2, w2, h2 = bbox

            # Normalize depth map to 0–255 if needed
            depth_visual = unknown_detector._normalize_to_0_255(sample_depth_map)
            depth_visual = cv2.cvtColor(depth_visual, cv2.COLOR_GRAY2BGR)
            contour_img = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2BGR)
            if exists:
                cv2.rectangle(depth_visual, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
                cv2.rectangle(contour_img, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
            else:
                cv2.rectangle(depth_visual, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)
                cv2.rectangle(contour_img, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)

            # Draw Boundaries
            contour_img_boundaries = draw_boundaries(contour_img, unknown_edge_margin)

            # Horizontally stack them
            combined1 = np.hstack((frame, depth_visual))
            black_img = np.zeros_like(frame)
            combined2 = np.hstack((contour_img, contour_img_boundaries))
            # Vertically stack them
            combined = np.vstack((combined1, combined2))

            # Show the combined video
            cv2.imshow("Test Unknown Detection", combined)
            # Press 'q' to quit
            if cv2.waitKey(1) == 27:
                break
            # Press 'q' to quit
            if cv2.waitKey(1) == 27:
                break
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()



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

if __name__ == '__main__':
    logger = setup_logging()

    # Initialize ConfigManager
    config_manager = ConfigManager.initialize(
        main_config_path="../configs/main_config.yaml",
        dataset_config_path="../configs/dataset_config.yaml",
        models_config_path="../configs/models_config.yaml"
    )

    # Load configurations
    main_config = config_manager.get_config("main")
    models_config = config_manager.get_config("models")
    dataset_config_path = "../configs/dataset_config.yaml"
    classes_names, mobile_classes_names, road_judging_classes = read_dataset_conf(dataset_config_path)

    # Initialize modules with injected dependencies and configurations
    object_detector = YoloDetector(
        model_path=models_config["yolo_model_path"],
        conf_threshold=models_config["yolo_inference_conf"],
        device=models_config["midas_device"] # Using midas_device for YOLO as well for consistency
    )
    depth_estimator = MidasEstimator(
        model_type=models_config["midas_model_type"],
        device=models_config["midas_device"]
    )
    object_tracker = ObjectTracker(
        classes_names=classes_names,
        mobile_classes_names=mobile_classes_names
    )
    scene_filter = SceneFilter()
    unknown_detector = UnknownDetector()
    scene_describer = SceneDescriber()
    speech_synthesizer = GTTSSpeechSynthesizer()

    # Initialize and run the PipelineManager
    pipeline_manager = PipelineManager(
        config_manager=config_manager,
        object_detector=object_detector,
        depth_estimator=depth_estimator,
        object_tracker=object_tracker,
        scene_filter=scene_filter,
        unknown_detector=unknown_detector,
        scene_describer=scene_describer,
        speech_synthesizer=speech_synthesizer,
        logger=logger
    )

    pipeline_manager.run()



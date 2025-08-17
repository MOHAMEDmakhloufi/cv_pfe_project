import cv2
import time
import numpy as np
from src.utils import has_reached_sequence_window, add_text, combine_and_show, create_centered_text_image

class PipelineManager:
    def __init__(self, config_manager, object_detector, depth_estimator, object_tracker, scene_filter, unknown_detector, scene_describer, speech_synthesizer, logger):
        self.config = config_manager
        self.detector = object_detector
        self.depth_estimator = depth_estimator
        self.tracker = object_tracker
        self.scene_filter = scene_filter
        self.unknown_detector = unknown_detector
        self.scene_describer = scene_describer
        self.speech_synthesizer = speech_synthesizer
        self.logger = logger

        self.main_config = self.config.get_config("main")
        self.dataset_config = self.config.get_config("dataset")
        self.models_config = self.config.get_config("models")

        self.frames_processed_per_seconds = self.main_config.get("frames_processed_per_seconds", 15)
        self.seconds_per_sequence = self.main_config.get("seconds_per_sequence", 3)
        self.unknown_edge_margin = self.main_config.get("unknown_detection", {}).get("edge_margin", 0.1)
        self.unknown_box_h_threshold = self.main_config.get("unknown_detection", {}).get("box_h_threshold", 10)

        self.depth_threshold_static_obj = self.main_config.get("depth_threshold_static_obj", 0.7)
        self.depth_threshold_mobile_obj = self.main_config.get("depth_threshold_mobile_obj", 0.5)

        self.classes_names = self.dataset_config.get("names", [])
        self.mobile_classes_names = self.dataset_config.get("mobile", [])
        self.road_judging_classes = self.dataset_config.get("road_judging_objects", [])

    def run(self):
        source = self.main_config.get("video_source")
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            self.logger.error(f"Error: Could not open video source: {source}")
            return

        fps = round(cap.get(cv2.CAP_PROP_FPS))
        self.logger.info(f"Video FPS: {fps}")

        frames_per_sequence = self.seconds_per_sequence * fps
        frames_processed_per_sequence = self.seconds_per_sequence * self.frames_processed_per_seconds
        step = round(max(frames_per_sequence, frames_processed_per_sequence) / frames_processed_per_sequence, 0)
        self.logger.info(f'Frame Processing Step: {step}')

        frame_count = 0
        processed_frame_count = 0
        processed_sequence_count = 0
        frames_list = []
        tracked_objects = dict()
        time_execution_count = time.time()
        combined1, combined2, combined3 = None, None, None
        try:
            while True:
                ret, frame = cap.read()

                if frame_count == 0:
                    combined2 = np.hstack((np.zeros_like(frame), np.zeros_like(frame), np.zeros_like(frame)))
                    combined3 = np.hstack((np.zeros_like(frame), np.zeros_like(frame), np.zeros_like(frame)))
                if not ret:
                    self.logger.info("End of video stream.")
                    break

                if frame_count % frames_per_sequence == 0 and frame_count != 0 and has_reached_sequence_window(time_execution_count, self.seconds_per_sequence):
                    self.logger.info(f"\n*** Sequence NB : {processed_sequence_count+1} ***\n")
                    self.logger.info(f'Processed {processed_frame_count} frames in sequence.')
                    # Parse objects information on full sequence
                    self.tracker.parse_sequence_tracking_results(tracked_objects, self.seconds_per_sequence)
                    self.logger.info(f'After Parsed Objects: {len(tracked_objects.keys())}')

                    # Estimate Depth from Optimal Frames
                    frames_depth_estimation = self.depth_estimator.estimate_depth_from_optimal_frames(frames_list, tracked_objects)

                    # --- Depth estimation display ---
                    last_frame_key = sorted(frames_depth_estimation.keys())[-1]
                    copied_frame = frames_list[last_frame_key].copy()
                    visual_depth = self.depth_estimator.get_visual_depth_map(frames_depth_estimation[last_frame_key])
                    add_text(copied_frame, f'Frame {last_frame_key}')
                    add_text(visual_depth, f'Depth Map of Frame {last_frame_key}')
                    header_frame = create_centered_text_image(frame.shape,
                                                              f'Example of Depth Estimation\nSequence {processed_sequence_count + 1}, Frame {last_frame_key}.')
                    combined2 = np.hstack((header_frame, copied_frame, visual_depth))
                    combine_and_show(combined1, combined2, combined3)

                    self.logger.info(f'Depth estimation frames: {len(frames_depth_estimation.keys())} / {processed_frame_count}')

                    # Filter out mobile objects whose trajectories do not intersect with the path of the blind person
                    tracked_objects = self.scene_filter.filter_mobile_objects_by_direction(tracked_objects, frame.shape)
                    self.logger.info(f'After Filtered out mobile objects whose trajectories do not intersect with the path of the blind person: {len(tracked_objects.keys())}')

                    # Filter near objects using midas depth estimation
                    tracked_objects = self.scene_filter.filter_static_objects_by_depth(tracked_objects, frames_depth_estimation, frame.shape,
                                                                                         edge_margin=self.unknown_edge_margin, depth_threshold=self.depth_threshold_static_obj)
                    self.logger.info(f'After Filtered near static objects: {len(tracked_objects.keys())}')
                    tracked_objects = self.scene_filter.filter_mobile_objects_by_depth(tracked_objects, frames_depth_estimation, depth_threshold=self.depth_threshold_mobile_obj)
                    self.logger.info(f'After Filtered near mobile objects: {len(tracked_objects.keys())}')

                    # Detect unknowns if tracked_objects is empty
                    if len(tracked_objects.keys()) == 0:
                        self.logger.info(f'No tracked objects. Checking for unknown objects.')
                        if len(frames_depth_estimation) < 3:
                            last_frames_for_unknown = frames_list[-5:] # Use last few frames for unknown detection if not enough depth frames
                            frames_depth_estimation = self.depth_estimator.estimate_depth_from_frames({i: value for i, value in enumerate(last_frames_for_unknown, start=len(frames_list)-len(last_frames_for_unknown))})

                        exist_unknown_object, bbox, contour_img, frame_id = self.unknown_detector.has_unknown_objects_in_depth_frames(frames_depth_estimation, self.unknown_edge_margin, self.unknown_box_h_threshold)
                        if exist_unknown_object:
                            contour_img = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2BGR)
                            visual_depth = self.depth_estimator.get_visual_depth_map(
                                frames_depth_estimation[frame_id])
                            x2, y2, w2, h2 = bbox
                            cv2.rectangle(contour_img, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
                            add_text(visual_depth, f'Depth Map of Frame {last_frame_key}')
                            add_text(contour_img, f'Contour Image of Depth Frame {last_frame_key}')
                            header_frame = create_centered_text_image(frame.shape,
                                                                      f'Unknown Object Detection\nSequence {processed_sequence_count + 1}, Depth Frame {frame_id}.')
                            combined3 = np.hstack((header_frame, visual_depth, contour_img))
                            combine_and_show(combined1, combined2, combined3)
                        sentence = self.scene_describer.generate_scene_description_for_unknown(exist_unknown_object)
                    else:
                        sentence = self.scene_describer.generate_scene_description(tracked_objects, self.road_judging_classes)

                    self.logger.info(f'Generated scene description: {sentence if sentence else "The path ahead is clear."}')
                    if sentence:
                        self.speech_synthesizer.speak(sentence)

                    processed_frame_count = 0
                    processed_sequence_count += 1
                    frames_list.clear()
                    tracked_objects.clear()
                    frames_depth_estimation.clear()
                    time_execution_count = time.time()

                elif frame_count % step == 0:
                    self.logger.debug(f'Current processing frame: {processed_frame_count}')

                    # Run YOLO detection
                    yolo_tracking_result = self.detector.object_tracking(frame)
                    self.tracker.parse_tracking_result(yolo_tracking_result, tracked_objects, processed_frame_count)

                    frames_list.append(frame)
                    processed_frame_count += 1

                    # --- YOLO tracking display ---
                    annotated_frame = yolo_tracking_result.plot()
                    copied_frame = frame.copy()
                    add_text(copied_frame, f'Frame {processed_frame_count}')
                    add_text(annotated_frame, f'Annotated Frame {processed_frame_count}')
                    header_frame = create_centered_text_image(frame.shape, f'Sequence {processed_sequence_count+1}:\n Original and Annotated Frames.' )
                    combined1 = np.hstack((header_frame, copied_frame, annotated_frame))
                    combine_and_show(combined1, combined2, combined3)


                frame_count += 1
                # Press 'q' to quit
                if cv2.waitKey(1) == 27:
                    break
        except Exception as e:
            self.logger.error(f"An error occurred during pipeline execution: {e}")
        finally:
            self.speech_synthesizer.shutdown()
            cap.release()
            cv2.destroyAllWindows()



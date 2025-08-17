import time

import yaml
import numpy as np
import cv2

def read_yaml_config(file_path, key=None):
    """Reads a YAML configuration file and returns the content.

    Args:
        file_path (str): The path to the YAML configuration file.
        key (str, optional): If provided, returns the value associated with this key.
                             Otherwise, returns the entire config.

    Returns:
        dict or list or any: The content of the YAML file, or the value associated with the key.
                              Returns an empty dict/list or None if the file is not found,
                              is not valid YAML, or the key is not found.
    """
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
            if key:
                return config.get(key)
            return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {file_path}")
        return None if key else {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {file_path}: {e}")
        return None if key else {}
    except Exception as e:
        print(f"An unexpected error occurred while reading {file_path}: {e}")
        return None if key else {}

def read_dataset_conf(dataset_conf_path):
    classes_names = read_yaml_config(dataset_conf_path, 'names')
    mobile_classes_names = read_yaml_config(dataset_conf_path, 'mobile')
    road_judging_classes = read_yaml_config(dataset_conf_path, 'road_judging_objects')

    return classes_names or [], mobile_classes_names or [], road_judging_classes or []


def has_reached_sequence_window(start_time, target_duration, tolerance=1):
    elapsed = time.time() - start_time
    return elapsed >= target_duration - tolerance


def add_text( image, text, position=(50, 50), color=(0, 0, 255), font_scale=1, thickness=2):
    """Utility to draw bold colored text on an image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return image


def combine_and_show( combined1=None, combined2=None, combined3=None, window_name="Real-Time Scene Understanding"):
    """Stack given frames and display them in a single window."""
    rows = []
    if combined1 is not None:
        rows.append(combined1)
    if combined2 is not None:
        rows.append(combined2)
    if combined3 is not None:
        rows.append(combined3)

    if rows:
        combined = np.vstack(rows)
        cv2.imshow(window_name, combined)

def create_centered_text_image(shape, text):
    height, width = shape[0], shape[1]   # shape = (H, W)
    # Create a black image (3 channels)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (255, 255, 255)

    # Split text into lines
    lines = text.split("\n")

    # Get height of one line
    (line_width, line_height), baseline = cv2.getTextSize("Ag", font, font_scale, thickness)
    total_text_height = len(lines) * (line_height + baseline)

    # Starting Y so that block of text is vertically centered
    y0 = (height - total_text_height) // 2 + line_height

    # Draw each line
    for i, line in enumerate(lines):
        (text_width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        x = (width - text_width) // 2
        y = y0 + round(i * (line_height + baseline) * 1.5)
        cv2.putText(img, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return img
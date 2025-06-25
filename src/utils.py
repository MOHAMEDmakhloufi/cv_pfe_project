import yaml

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



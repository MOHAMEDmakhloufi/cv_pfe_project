import yaml

class ConfigManager:
    _instance = None

    def __new__(cls, main_config_path=None, dataset_config_path=None, models_config_path=None):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._configs = {}
            if main_config_path:
                cls._instance._configs['main'] = cls._instance._load_config(main_config_path)
            if dataset_config_path:
                cls._instance._configs['dataset'] = cls._instance._load_config(dataset_config_path)
            if models_config_path:
                cls._instance._configs['models'] = cls._instance._load_config(models_config_path)
        return cls._instance

    def _load_config(self, file_path):
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {file_path}")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {file_path}: {e}")
            return {}

    def get_config(self, config_name, key=None):
        if config_name not in self._configs:
            return None
        if key is None:
            return self._configs[config_name]
        return self._configs[config_name].get(key)

    @classmethod
    def initialize(cls, main_config_path, dataset_config_path, models_config_path):
        if cls._instance is None:
            cls._instance = cls(main_config_path, dataset_config_path, models_config_path)
        return cls._instance




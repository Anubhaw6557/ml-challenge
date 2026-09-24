import yaml
from pathlib import Path
from typing import Dict, Any

_config = None

def load_config(config_path: str = None) -> Dict[str, Any]:
    global _config
    if _config is not None:
        return _config
    
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    
    with open(config_path) as f:
        _config = yaml.safe_load(f)
    
    return _config

def get_config() -> Dict[str, Any]:
    if _config is None:
        return load_config()
    return _config

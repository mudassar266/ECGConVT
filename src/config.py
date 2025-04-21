import yaml

def load_config(path="config.yaml"):
    """Load training/eval configuration from a YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

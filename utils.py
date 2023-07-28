import yaml

def load_yaml(file_path):
    """ Wrapper to load a yaml file.

    Args:
        file_path: (str) path to the file to load.

    Returns:
        dict with loaded parameters.
    """

    with open(file_path, 'r') as f:
        file = yaml.safe_load(f)

    return file
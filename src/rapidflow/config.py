import os
from pathlib import Path


class Path:
    file_path = os.path.dirname(os.path.abspath(__file__))
    root_path = Path(file_path).parent.parent
    data_path = os.path.join(root_path, "data")
    data_network_path = os.path.join(data_path, "network_datasets")
    experiments_path = os.path.join(root_path, "experiments")

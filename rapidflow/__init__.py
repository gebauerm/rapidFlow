from pathlib import Path
import os

URL = "https://github.com/gebauerm/rapidFlow"
__version__ = "0.1.8c"

install_requires = [
        "optuna>=2.9.1",
        "click>=8.0.1", "scikit-learn", "scipy", "networkx>=2.5.1", "psycopg2-binary", "docker>=5.0.3",
        "pandas", "tqdm>=4.62.3"],

test_require = ["pytest==7.1.2", "pytest-cov==3.0.0"]

long_description = ""

this_directory = Path(__file__).parent.parent
if os.path.exists(this_directory/"README.md"):
    long_description = (this_directory / "README.md").read_text()
else:
    long_description = ""

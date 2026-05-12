import json
import sys
from pathlib import Path

# Get the absolute path of the directory containing loader.py (src/loader/)
current_file_path = Path(__file__).resolve()

# Go up two levels to reach 'OptML_CNN_project'
# parent 0: src/loader/, parent 1: src/, parent 2: OptML_CNN_project/
ROOT_DIR = current_file_path.parents[2]

# Optional: Add ROOT_DIR to sys.path so tasks' and 'src' imports work everywhere
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

SRC = ""

def parser():
    with open(ROOT_DIR/'config.json') as file:
        parsed = json.load(file)
        print(parsed)

    return parsed

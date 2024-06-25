import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(dir_path)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
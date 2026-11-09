"""Lets every script import scene_repair when run as `python scripts/xyz.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

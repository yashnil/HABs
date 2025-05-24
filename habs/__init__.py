# habs/__init__.py  – keep it tiny!
"""
HABS research pipeline top-level package.
"""

# Optionally make the most-used sub-modules easy to reach:
from importlib import import_module      # noqa: E402

feature_engineering = import_module("habs.feature_engineering")
preprocess          = import_module("habs.preprocess")
quality_control     = import_module("habs.quality_control")

__all__ = ["feature_engineering", "preprocess", "quality_control"]
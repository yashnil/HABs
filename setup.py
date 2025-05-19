# setup.py
from setuptools import setup, find_packages

setup(
    name="habs",
    version="0.1",
    packages=find_packages(include=["scripts*", "preprocess*", "preprocessing*"]),
)
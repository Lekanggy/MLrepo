# setup.py
from setuptools import setup, find_packages

setup(
    name="my_tools",  # Name of the module
    version="0.1",  # Version number
    packages=find_packages(),  # Automatically find packages
    install_requires=["pandas", "numpy"],  # List of dependencies (if any)
    author="Idris",
    author_email="lekanggy12@gmail.com",
    description="A simple Python module",
    url="https://github.com/Lekanggy/my_tools",
)
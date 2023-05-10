import os
import sys
from setuptools import setup, find_packages
from sys import platform

PACKAGE_NAME = "CloudReg"
DESCRIPTION = "Automatic terabyte-scale cross-modal brain volume registration"
with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()
AUTHOR = (
    "Vikram Chandrashekhar, Daniel Tward",
)
URL = "https://github.com/neurodata/CloudReg"
REQUIRED_PACKAGES = [
    "paramiko",
    "joblib",
    "SimpleITK>=2.0.0",
    "boto3",
    "tqdm",
    "awscli",
    "cloud-volume",
    "h5py",
    "scipy",
    "tinybrain",
    "tifffile",
    "imagecodecs",
    # "mpi4py"
]

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(PROJECT_PATH, "cloudreg", "__init__.py")):
    if line.startswith("__version__ = "):
        VERSION = line.strip().split()[2][1:-1]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    install_requires=REQUIRED_PACKAGES,
    url=URL,
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
    include_package_data=True,
)

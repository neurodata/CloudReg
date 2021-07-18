# CloudReg
[![Documentation Status](https://readthedocs.org/projects/cloudreg/badge/?version=latest)](https://cloudreg.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/204720122.svg)](https://zenodo.org/badge/latestdoi/204720122)

## `CloudReg` is a tool for cross-modal, nonlinear, image registration between arbitrary image volumes.

- [Overview](#overview)
- [Documentation](#documentation)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [License](#license)
- [Issues](#issues)
- [Citing `CloudReg`](#citing-cloudreg)


# Overview
Quantifying terascale multi-modal human and animal imaging data requires scalable analysis tools. We developed CloudReg, an automated, terascale, cloud-based image analysis pipeline for preprocessing and cross-modal, non-linear registration between volumetric datasets with artifacts. CloudReg was developed using cleared mouse and rat brain light-sheet microscopy images, but is also accurate in registering the following datasets to their respective atlases: *in vivo* human and *ex vivo* macaque brain magnetic resonance imaging, *ex vivo* murine brain micro-computed tomography. Our extensive documentation (below) can enable deployment of this tool for many other datasets/research questions.

# Documentation
The official documentation with usage is at https://cloudreg.neurodata.io

Please visit the [Run section](https://cloudreg.neurodata.io/run.html) in the official website for more in depth usage.

# System Requirements
## Hardware requirements
`CloudReg` requires only a standard computer with enough RAM to support the in-memory operations since the majority of work is run in cloud services.

## Software requirements
### OS Requirements
`CloudReg` is tested on the following OSes and requires Python 3:
- Linux x64
- macOS x64

# Installation Guide
Please see [Setup](https://cloudreg.neurodata.io/setup.html) on the official website for detailed set up information. 

# License
This project is covered under the Apache 2.0 License.

# Issues
We appreciate detailed bug reports and feature requests (though we appreciate pull requests even more!). Please visit our [issues](https://github.com/neurodata/CloudReg/issues) page if you have questions or ideas.

# Citing `CloudReg`
If you find `CloudReg` useful in your work, please cite the tool via the [CloudReg paper](https://www.nature.com/articles/s41592-021-01218-z#citeas)

> Chandrashekhar, V., Tward, D.J., Crowley, D. et al. CloudReg: automatic terabyte-scale cross-modal brain volume registration. Nat Methods (2021). https://doi.org/10.1038/s41592-021-01218-z

..  -*- coding: utf-8 -*-

.. _contents:

Overview of CloudReg
====================

Motivation
----------
Quantifying terascale multi-modal human and animal imaging data requires scalable analysis tools. We developed CloudReg, an automated, terascale, cloud-based image analysis pipeline for preprocessing and cross-modal, non-linear registration between volumetric datasets with artifacts. CloudReg was developed using cleared murine brain light-sheet microscopy images, but is also accurate in registering the following datasets to their respective atlases: in vivo human and ex vivo macaque brain magnetic resonance imaging, ex vivo murine brain micro-computed tomography.


Results
-------
.. figure:: 
   :scale: 50
   :alt: CLARITY registered to Allen Reference Atlas

   This is the caption of the figure (a simple paragraph).

Documentation
=============

CloudReg is a pipeline for terascale image preprocessing and 3D nonlinear registration between two image volumes wth polynomial intensity correspondence.

.. toctree::
   :maxdepth: 1

   setup
   run
   reference/index
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

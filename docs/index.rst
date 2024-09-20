|Logo| ``betaburst``: detect beta bursts in brain signals, specifically EEG and MEG data.
============================================

Introduction
------------

**BetaBursts** is a Python package designed to detect beta bursts in brain signals, specifically EEG and MEG data. Beta bursts are rapid oscillations in the beta frequency band (13-30 Hz) and are crucial for movement-related cortical dynamics. This package provides a reliable method for detecting these bursts using advanced signal processing techniques.

The method is described in the paper by [Maciek Szul](http://www.isc.cnrs.fr/index.rvt?member=maciek%5F%5Fszul) et al. (2023) "Diverse beta burst waveform motifs characterize movement-related cortical dynamics" Progress in Neurobiology. https://doi.org/10.1016/j.pneurobio.2023.102490

The code have been developed by Maciek Szul, [Sotirios Papadopoulos](http://www.isc.cnrs.fr/index.rvt?member=sotiris%5Fpapadopoulos), [Ludovic DARMET](http://www.isc.cnrs.fr/index.rvt?language=en&member=ludovic%5Fdarmet) and [Jimmy Bonaiuto](http://www.isc.cnrs.fr/index.rvt?member=james%5Fbonaiuto), head of the [DANC lab](https://www.danclab.com/).

Available modules
-----------------

Here is a list of the modules available in ``betaburst``:

.. currentmodule:: betaburst

.. toctree::
   :maxdepth: 1

.. autosummary::
   :caption: betaburst
   :toctree: _autosummary

   betaburst.detection
   betaburst.analysis
   betaburst.superlet
   betaburst.utils


Tutorials
----------------

A collection of tutorials is available:

.. toctree::
   :maxdepth: 2

   auto_tutorials/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
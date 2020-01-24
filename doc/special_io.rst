special_io module doc's
#######################

This module contains special classes and functions to handle the io for the thermography experiments.

.. module:: ipychord3.special_io

.. contents::

The HDF5 file structure for collection of data
----------------------------------------------

The HDF5 file contains the thermogrphy, scatterering and mechanical data from the experiments.
The thermography and mechanical data is interpolated on the timesstamp of the scattering pattern.
The structure is given in this :download:`pdf<data/hdf5_structure.pdf>` (:download:`source<data/hdf5_structure.graphml>` for `yEd
<https://www.yworks.com/yed>`_)


Overview
--------

**Read data**

Functions/Classes to read data

===================================  ==========================================
:class:`.IRBISTextReader`            Read IRBIS text files of images
:class:`.PILCReader`                 Read data recorded with PiLC
===================================  ==========================================

**Synchronization**

Functions/Classes to synchronize recorded data

===================================  ==========================================
:func:`.cbf_pattern_selector`        Select CBFs with non-zero average
:func:`.pilc_data_for_pattern`       Select PiLC data for a set of pattern
:class:`.ThermoSelector`             Select thermography images
===================================  ==========================================


**Data collection**

Functions/Classes to handle the collection of data in the hdf5 file introduced above.

===================================  =================================================
:class:`.DataCollection`             Class to access the data
:class:`.DataSubCollection`          Class to represent a part of the data collection
:class:`.DataCollectionItem`         Class to represent one item of the DataCollection
===================================  =================================================


API documentation
-----------------

Read data
^^^^^^^^^

.. autoclass:: ipychord3.special_io.IRBISTextReader
    :members:
.. autoclass:: ipychord3.special_io.PILCReader
    :members:


Synchronization
^^^^^^^^^^^^^^^

.. autofunction:: ipychord3.special_io.cbf_pattern_selector
.. autofunction:: ipychord3.special_io.pilc_data_for_pattern
.. autoclass:: ipychord3.special_io.ThermoSelector
    :members:

Data collection
^^^^^^^^^^^^^^^

.. autoclass:: ipychord3.special_io.DataCollection
    :members:
.. autoclass:: ipychord3.special_io.DataSubCollection
    :members:
.. autoclass:: ipychord3.special_io.DataCollectionItem
    :members:

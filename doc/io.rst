io module doc's
###############

.. module:: ipychord3.io

.. contents::

The HDF5 file structure
-----------------------

- The file root has the attributes:

    :HDF5_Version: the version of the HDF5 library used
    :h5py_version: the version of h5py used
    :file_time: timestamp of the file creation

- The file has one group called `entry` which contains the actual data.

    - All the parameters are saved as attributes to `entry`.
    - The image is saved as dataset under `/entry/map`.


Overview
--------

**Read detector data**

Functions/Classes to read data provieded by the detector

===================================  ==========================================
:class:`.FIOreader`                  Read ``.fio`` files
:class:`.CBFreader`                  Read ``.cbf`` files
:class:`.LambdaReader`               Read ``.nxs`` files from Lambda detector
===================================  ==========================================

**Read/write processed pattern**

Functions/Classes to read and write processed patterns

===================================  ==========================================
:func:`.writeh5`                     Save pattern as HDF5 files
:func:`.readh5`                      Read read pattern from HDF5 file
===================================  ==========================================

**Miscellaneous**

===================================  ======================================================
:class:`.NexusMask`                  Handle the ``pixel_mask`` of a Nexus file
===================================  ======================================================

API documentation
-----------------

Read detector data
^^^^^^^^^^^^^^^^^^

.. autoclass:: ipychord3.io.FIOreader
    :members:
.. autoclass:: ipychord3.io.CBFreader
    :members:
.. autoclass:: ipychord3.io.LambdaReader
    :members:


Read/write processed pattern
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: ipychord3.io.writeh5
.. autofunction:: ipychord3.io.readh5

Miscellaneous
^^^^^^^^^^^^^

.. autoclass:: ipychord3.io.NexusMask
    :members:


utils module doc's
##################

This module contains some useful utility functions and classes.

.. module:: ipychord3.utils

.. contents::


Overview
--------

**Coordinate transforms**

===================================  ===========================================================
:func:`.to_polar`                    Convert x,y to r,phi
:func:`.to_cartesian`                Convert r,phi to x,y
===================================  ===========================================================

**Baseline fitting**

===================================  ===========================================================
:func:`.baseline_als`                Baseline fitting using asymmetric least squares smoothing
===================================  ===========================================================

**Path sorting**

===================================  ===========================================================
:func:`.sortedScanPaths`             Sort paths for an online scan
===================================  ===========================================================

**Data selection and modification**

===================================  ===========================================================
:func:`.indices_below`               Get indices below a certain value
:func:`.indices_above`               Get indices above a certain value
:func:`.remove_around`               Remove data around an index
:func:`.invert_indices`              
===================================  ===========================================================

**Test data**

===================================  ===========================================================
:func:`.gaussian2d`                  2d Gaussian in cartesian coordinates
:func:`.gaussian2d_polar`            2d Gaussian in polar coordinates
:class:`.TestPattern`                Create a test pattern
===================================  ===========================================================

API documentation
-----------------

Coordinate transforms
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: ipychord3.utils.to_polar
.. autofunction:: ipychord3.utils.to_cartesian

Baseline fitting
^^^^^^^^^^^^^^^^

.. autofunction:: ipychord3.utils.baseline_als

Path sorting
^^^^^^^^^^^^

.. autofunction:: ipychord3.utils.sortedScanPaths

Data selection and modification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: ipychord3.utils.indices_below
.. autofunction:: ipychord3.utils.indices_above
.. autofunction:: ipychord3.utils.remove_around
.. autofunction:: ipychord3.utils.invert_indices

Test data
^^^^^^^^^

.. autofunction:: ipychord3.utils.gaussian2d
.. autofunction:: ipychord3.utils.gaussian2d_polar
.. autoclass:: ipychord3.utils.TestPattern
    :members:
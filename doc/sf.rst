sf module doc's
###############

.. module:: ipychord3.sf

.. contents::

Overview
--------

**Interactive functions**

Functions for interactive use.

===================================  =================================================================================================
:func:`.show`                        Show a pattern
:func:`.debye`                       Draw the Debye-Scherrer rings
:func:`.killcircle`                  Select circle in pattern and mask it
:func:`.killbox`                     Select rectangles in pattern and mask them
:func:`.killpoly`                    Select polygons in pattern and mask them
:func:`.kill_ringsector`             Select ring sectors in pattern and mask them
:func:`.rotate_pattern`              Rotate the pattern
:func:`.create_peak_masks`           Select ring sectors in figure to select peaks and a corresponding reference.
===================================  =================================================================================================

**Modifier functions**

Functions to edit the pattern

===================================  ============================================================================================================
:func:`.mask_circles`                Mask circles in a pattern
:func:`.mask_rectangles`             Mask rectangles in a pattern
:func:`.mask_polygons`               Mask polygons in a pattern
:func:`.mask_ring_sectors`           Mask ring sectors in a pattern
:func:`.make_peak_mask`              Create masks for peaks
:func:`.extend_image`                Extend the pattern
:func:`.harmony`                     Harmonize the pattern by exploiting symmetry
:func:`.harmony2`                    Harmonize the pattern by exploiting rotational symmetry by 180°
:func:`.local_median_filter`         Apply a median filter on a part of the pattern
===================================  ============================================================================================================

**Interaction classes**

These classes define selections for cicles, rectangles ... and other interactive modifications

===================================  ===================================
:class:`.InteractBase`               Base class for interaction classes
:class:`.SelectCircles`              select circles in fig
:class:`.SelectRectangles`           select rectangles in fig
:class:`.SelectPolygons`             select polygons in fig
:class:`.SelectRingSectors`          select ring sectors in fig
:class:`.SelectTwoRingSectors`       select two ring sectors in fig
:class:`.RotatePattern`              rotate the pattern
===================================  ===================================

**Miscellaneous**

===================================  =================================================================================
:class:`.Circle`                     A circle with ``center`` and ``radius``
:class:`.Rectangle`                  A rectangle with ``corner``, ``width`` and ``height``
:class:`.Polygon`                    A polygon with ``vertices``
:class:`.RingSector`                 A ring sector with ``theta1``, ``theta2``, ``radius``, ``width``, ``center``
:func:`.handle_close`                Stop event loop
:func:`.midpnt`                      Compute center pixel of the pattern
===================================  =================================================================================


API Documentation
-----------------

Interactive functions
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: ipychord3.sf.show
.. autofunction:: ipychord3.sf.debye
.. autofunction:: ipychord3.sf.killcircle
.. autofunction:: ipychord3.sf.killbox
.. autofunction:: ipychord3.sf.killpoly
.. autofunction:: ipychord3.sf.kill_ringsector
.. autofunction:: ipychord3.sf.rotate_pattern
.. autofunction:: ipychord3.sf.create_peak_masks

Modifier functions
^^^^^^^^^^^^^^^^^^

.. autofunction:: ipychord3.sf.mask_circles
.. autofunction:: ipychord3.sf.mask_rectangles
.. autofunction:: ipychord3.sf.mask_polygons
.. autofunction:: ipychord3.sf.mask_ring_sectors
.. autofunction:: ipychord3.sf.make_peak_mask
.. autofunction:: ipychord3.sf.extend_image
.. autofunction:: ipychord3.sf.harmony
.. autofunction:: ipychord3.sf.harmony2
.. autofunction:: ipychord3.sf.local_median_filter

Interaction classes
^^^^^^^^^^^^^^^^^^^

.. autoclass:: ipychord3.sf.InteractBase
    :members:
.. autoclass:: ipychord3.sf.SelectCircles
    :members:
.. autoclass:: ipychord3.sf.SelectRectangles
    :members:
.. autoclass:: ipychord3.sf.SelectPolygons
    :members:
.. autoclass:: ipychord3.sf.SelectRingSectors
    :members:
.. autoclass:: ipychord3.sf.SelectTwoRingSectors
    :members:
.. autoclass:: ipychord3.sf.RotatePattern
    :members:

Miscellaneous
^^^^^^^^^^^^^

.. autoclass:: ipychord3.sf.Circle
    :members:
.. autoclass:: ipychord3.sf.Rectangle
    :members:
.. autoclass:: ipychord3.sf.Polygon
    :members:
.. autoclass:: ipychord3.sf.RingSector
    :members:
.. autofunction:: ipychord3.sf.handle_close
.. autofunction:: ipychord3.sf.midpnt

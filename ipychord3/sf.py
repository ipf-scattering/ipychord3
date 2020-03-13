# Copyright (c) 2019 ipychord3 authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""This module contains some of the old function with sf_ prefix
"""

import logging

from copy import deepcopy
from collections import namedtuple

import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Wedge
from matplotlib.path import Path

import skimage.draw as sidraw
import skimage.transform as sitransform

import numpy as np

from scipy import ndimage
from scipy.ndimage.filters import median_filter

from . import cmaps


# setup logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# we can't use TK backend, it will crash with python 3.4 on windows:
# https://github.com/ipython/ipython/issues/8921#issuecomment-151046708
# matplotlib.use('TkAgg')
# matplotlib.use('Qt4Agg')


# ----------------------------------------------------------------------------
# Miscellaneous
# ----------------------------------------------------------------------------


Circle = namedtuple('Circle', ['center', 'radius'])
Rectangle = namedtuple('Rectangle', ['corner', 'width', 'height'])
Polygon = namedtuple('Polygon', ['vertices'])
RingSector = namedtuple('RingSector', ['theta1', 'theta2', 'radius',
                                       'width', 'center'])


def handle_close(event):
    """Handle the closing of a figure,

    it stops the event loop so the program can continue
    """
    fig = event.canvas.figure
    logger.debug("stopping blocking event loop")
    fig.canvas.stop_event_loop()


def prepare_patter_for_show(img, ulev=None, dlev=None, log=0, med=None,
                            over=None, neg=False, mag=False, clip=True):
    """prepare pattern for show

     Scales the pattern and computes ``ulev`` and ``dlev``

     A copy of the pattern is returned.

    :param img: the image dictionary
    :type img: dict
    :param ulev: show the image in certain interval, ulev defines the upper level.
    :type ulev: float
    :param dlev: defines the down level
    :type dlev: float
    :param log: show the image in log scale coordinate

        :log==0: show it in linear coordinate
        :log==1: show it in log() coordinate
        :log==2: show it in log(log()) coordinate
    :type log: int
    :param med: use median filter to estimate ulev, ``3 < med < 15`` otherwise ``med=5``
    :type med: float
    :param over: overestimation of the scales
    :type over: float
    :param neg: show the negative side of the image
    :type neg: bool
    :param mag: show magnitude of image
    :type mag: bool
    :param clip: clip negative values
    :type clip: bool
    :return: scaled pattern,  ulev, dlev
    :rtype: pattern dict, float, float
    """
    img = deepcopy(img)

    if mag:
        img['map'] = np.absolute(img['map'])

    if neg:
        mask = img['map'] <= 0.0
        img['map'] *= mask
        img['map'] *= -1.0

    if clip:
        img['map'] = img['map'] * (img['map'] >= 0.0)

    if log == 0:
        logger.info("The image is shown in the linear coordinates")
        if ulev is None:
            if med is not None:
                if not 3 < med < 15:
                    med = 5
                ulev = ndimage.median_filter(img['map'], med).max()
                logger.debug("ulev is None: estimated with median %g linear as %g" %(med, ulev))
            else:
                ulev = img['map'].max()
                logger.debug("ulev is None: estimated directly as %g" %ulev)
            logger.debug("linear ulev = %g" %ulev)

        else:
            logger.debug("ulev set by user as %g" %ulev)

        if dlev is None:
            dlev = img['map'].min()
            logger.debug("dlev is None: estimated as %g" %dlev)
        else:
            logger.debug("dlev set as: %g" %dlev)
    elif log == 1:
        img['map'] = np.log(img['map']+1.0)
        if ulev is None:
            logger.debug("estimating ulev")
            ulev = (img['map']+1.0).max()
        dlev = (img['map']+1.0).min()
        logger.debug("log scale used: dlev = %g,  ulev = %g" % (dlev, ulev))
    elif log == 2:
        img['map'] = np.log(np.log(img['map']+1.0)+1.0)
        if ulev is None:
            ulev = (img['map']+1.0).max()
        dlev = (img['map']+1.0).min()
        logger.debug("double log scale used: dlev = %g, ulev = %g" % (dlev, ulev))

    if over is not None:
        ulev /= over
        logger.info("overestimated ulev corrected to %g" % ulev)

    return img, ulev, dlev

# ----------------------------------------------------------------------------
# Interactive functions
# ----------------------------------------------------------------------------


def show(img, ulev=None, dlev=None, log=0, med=None, win=None, block=False, show=True,
         cmap=cmaps.lut05(), over=None, neg=False, mag=False, clip=True, scalefig=1.2):
    """ show the image under different conditions

    .. note::  This function can not show the positive and negative value
               in one image except we use the absolute value to show all the
               values in one image.

    :param img: the image dictionary
    :type img: dict
    :param ulev: show the image in certain interval, ulev defines the upper level.
    :type ulev: float
    :param dlev: defines the down level
    :type dlev: float
    :param log: show the image in log scale coordinate

        :log==0: show it in linear coordinate
        :log==1: show it in log() coordinate
        :log==2: show it in log(log()) coordinate
    :type log: int
    :param med: use median filter to estimate ulev, ``3 < med < 15`` otherwise ``med=5``
    :type med: float
    :param win:
    :type win: matplotlib window, int
    :param block: show the image in the interactive way and block the command line
    :type block: bool
    :param show: show the figure
    :type show: bool
    :param cmap: colormap to be passed to ``imshow``, delauft ``ipychord3.cmaps.lut05()``
    :type cmap: matplotlib.colors.Colormap or string
    :param over: overestimation of the scales
    :type over: float
    :param neg: show the negative side of the image
    :type neg: bool
    :param mag: show magnitude of image
    :type mag: bool
    :param clip: clip negative values
    :type clip: bool
    :param float scalefig: scale the figure by factor

    :return: figure
    :rtype: matplotlib figure object
    """
    # protect the virgin img

    kwargs_for_prepare = {'ulev': ulev, 'dlev': dlev, 'log': log, 'med': med,
                          'over': over, 'neg': neg, 'mag': mag, 'clip': clip}

    img, ulev, dlev = prepare_patter_for_show(img, **kwargs_for_prepare)

    # create figure
    w, h = figaspect(img['map'])
    if win is None:
        fig = plt.figure(figsize=(scalefig*w, scalefig*h))
    else:
        fig = plt.figure(win, figsize=(scalefig*w, scalefig*h))

    logger.info("dlev = %g  ulev = %g" % (dlev, ulev))

    # create the axis and show the image
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    ax.imshow(img['map'], interpolation='nearest', vmin=dlev, vmax=ulev, cmap=cmap, origin='upper')
    ax.set_aspect('equal')
    ax.set_axis_off()

    fig.canvas.set_window_title(img['filename']+'_sf_show')

    fig._sf_kwargs_for_prepare = kwargs_for_prepare
    if not show:
        return fig
    elif block:
        # now we start an extra event loop for this figure
        # it will block the program until fig.canvas.stop_event_loop() is called
        fig.canvas.mpl_connect('close_event', handle_close)
        fig.show()
        logger.debug("starting blocking event loop")
        fig.canvas.start_event_loop(timeout=-1)
    else:
        fig.show()
        logger.debug("show non-blocking figure: starting event loop to force drawing of figure")
        fig.canvas.start_event_loop(timeout=.01)  # start extra event loop for a short time to
                                                  # force drawing of the figure
        logger.debug("show non-blocking figure: event loop exited")
    return fig


def killcircle(img, fig={}):
    """Select circles in figure and mask them in the pattern

    :param img: pattern dict
    :param fig: a dictionary with keyword arguments for ``sf.show``
        or a matplotlib figure

        .. note:: If you use ``sf.show`` the figure must be created using ``block=False``.

            If a dict is passed ``block`` will be set to ``False``.

    :returns: masked pattern, mask, circles

    :raises RuntimeError:  if no circles have been drawn

    .. hint::

        :Draw circle: left mouse button to set center, set radius by clicking left button again
        :Modify circle:
            use +/- keys to increase/decrease radius by 1 px

            use arrow-keys to move center
        :Delete circle: backspace
        :Select circle: use "ctrl+[0..6]" to select one of the first 7 circles
    """
    if isinstance(fig, dict):
        logger.debug('creating figure')
        fig['block'] = False
        fig = show(img, **fig)

    selector = SelectCircles(fig)

    circles = selector.circles

    imgmask, mask = mask_circles(img, circles)

    return imgmask, mask, circles


def killbox(img, fig={}):
    """Select rectangles in figure and mask the pattern

    :param img: pattern dict
    :param fig: a dictionary with keyword arguments for ``sf.show``
        or a matplotlib figure

        .. note:: If you use ``sf.show`` the figure must be created using ``block=False``.

            If a dict is passed ``block`` will be set to ``False``.

    :returns: masked pattern, mask, rectangles

    :raises RuntimeError:  if no rectangles have been drawn

    .. hint::

        :Draw rectangle: left mouse button to set corner, set other corner by clicking left button again
        :Modify circle:
            use +/-/*/_  keys to increase/decrease x/y by 1 px

            use arrow-keys to first corner
        :Delete rectangle: backspace
    """
    if isinstance(fig, dict):
        logger.debug('creating figure')
        fig['block'] = False
        fig = show(img, **fig)

    selector = SelectRectangles(fig)
    rectangles = selector.rectangles

    imgmask, mask = mask_rectangles(img, rectangles)

    return imgmask, mask, rectangles


def killpoly(img, fig={}):
    """Select polygons in figure and mask them in the pattern

    :param img: pattern dict
    :param fig: a dictionary with keyword arguments for ``sf.show``
        or a matplotlib figure

        .. note:: If you use ``sf.show`` the figure must be created using ``block=False``.

            If a dict is passed ``block`` will be set to ``False``.

    :returns: masked pattern, mask, polygons

    :raises RuntimeError:  if no polygons have been drawn

    .. hint::

        :Draw polygon: left mouse button to set vertices, set last vertex with right mouse button
        :Modify polygon:
            use `shift-backspace` to delete a vertex

        :Delete polygon: backspace
    """
    if isinstance(fig, dict):
        logger.debug('creating figure')
        fig['block'] = False
        fig = show(img, **fig)

    selector = SelectPolygons(fig)
    polygons = selector.polygons

    imgmask, mask = mask_polygons(img, polygons)

    return imgmask, mask, polygons


def kill_ringsector(img, fig={}, center=None):
    """Select ring sectors in figure and mask them in the pattern

    :param img: pattern dict
    :param fig: a dictionary with keyword arguments for ``sf.show``
        or a matplotlib figure

        .. note:: If you use ``sf.show`` the figure must be created using ``block=False``.

            If a dict is passed ``block`` will be set to ``False``.

    :returns: masked pattern, mask, masks, sectors

    :raises RuntimeError:  if no sectors have been drawn

    .. hint::

        :Draw sector: left mouse button to set vertices, adjust position with keyboard (see ctrl-h), press space to draw new sector

        :Delete sector: backspace
    """
    if isinstance(fig, dict):
        logger.debug('creating figure')
        fig['block'] = False
        fig = show(img, **fig)

    selector = SelectRingSectors(fig, center=center)
    sectors = selector.sectors

    imgmask, mask, masks = mask_ring_sectors(img, sectors)

    return imgmask, mask, masks, sectors


def create_peak_masks(img, fig={}, center=None):
    """Select ring sectors in figure to select peaks and a corresponding reference.

    This function returns a stacked list of masks [peak_mask, reference_mask]

    :param img: pattern dict
    :param fig: a dictionary with keyword arguments for ``sf.show``
        or a matplotlib figure

        .. note:: If you use ``sf.show`` the figure must be created using ``block=False``.

            If a dict is passed ``block`` will be set to ``False``.

    :returns: masks, sectors

    :raises RuntimeError:  if no sectors have been drawn

    .. hint::

        :Draw sector: left mouse button to set vertices, adjust position with keyboard (see ctrl-h), press space to draw new sector.

        :Delete sector: backspace
    """
    if isinstance(fig, dict):
        logger.debug('creating figure')
        fig['block'] = False
        fig = show(img, **fig)

    selector = SelectTwoRingSectors(fig, center=center)
    sectors = selector.sectors

    mask = make_peak_mask(img, sectors)

    return mask, sectors


def rotate_pattern(img, fig={}, angle=None):
    """ Rotates the pattern by interactively by 0.3° / 1° or non-interactive by ``angle``

    :param img: pattern dict
    :param fig: a dictionary with keyword arguments for ``sf.show``
        or a matplotlib figure

        .. note:: If you use ``sf.show`` the figure must be created using ``block=False``.

            If a dict is passed ``block`` will be set to ``False``.
    :param angle: if not ``None`` rotate pattern by ``angle`` without opening a figure window

    :returns: rotated pattern, angle

    .. hint::

        :rotate clockwise:  ``r``: 0.3°  ``R``: 1°
        :rotate anticlockwise:  ``a``: 0.3°  ``A``: 1°
    """
    img = deepcopy(img)

    if angle is not None:
        img['map'] = ndimage.rotate(img['map'], angle, mode='constant', cval=0.0)
        img['beam_position'] = midpnt(img)
    else:
        if isinstance(fig, dict):
            logger.debug('creating figure')
            fig['block'] = False
            fig = show(img, **fig)

        rot = RotatePattern(fig, img)
        img = rot.img
        angle = rot.angle

    return img, angle


def debye(img, fig, center=None):
    """Draw the Debye-Scherrer rings, calculates diameter and center of each
    and returns the mean center for mirroring

    :param fig: a dictionary with keyword arguments for ``sf.show``
        or a matplotlib figure

        .. note:: If you use ``sf.show`` the figure must be created using ``block=False``.

            If a dict is passed ``block`` will be set to ``False``.

    :param center: set the beam position (x, y)

    :returns: ``[center x, center y, circles]``

    :raises RuntimeError:  if no circles have been drawn
    """

    if isinstance(fig, dict):
        logger.debug('creating figure')
        fig['block'] = False
        fig = show(img, **fig)

    print('Draw circles in Debye-Scherrer rings to calculate their diameter \
        and center.')
    selector = SelectCirclesSameCenter(fig, center)

    centers = []
    circles = selector.circles

    if not circles:
        raise RuntimeError("You have to create at least one circle")

    logger.debug("length of circles = %d" % len(circles))

    for i, circle in enumerate(circles):
        d = 2 * circle.radius
        center = circle.center
        centers.append(center)
        print("Circle %d: (%.4f, %.4f)@%.4f" % (i + 1, center[0], center[1], d))

    return circles[0].center[0], circles[0].center[1], circles


# ----------------------------------------------------------------------------
# Modifier functions
# ----------------------------------------------------------------------------

def mask_circles(img, circles):
    """mask ``circle`` in ``img``

    :param img: pattern dict
    :param circles: list of ``sf.Circle``'s

    :returns: masked image, mask
    """

    imgmask = deepcopy(img)

    mask = np.ones_like(imgmask['map'], dtype=np.uint8)

    for circle in circles:
        temp_mask = np.ones_like(imgmask['map'], dtype=np.uint8)
        x, y = circle.center
        rr, cc = sidraw.circle(y, x, circle.radius, shape=imgmask['map'].shape)
        temp_mask[rr, cc] = 0
        mask *= temp_mask

    imgmask['map'] = mask * imgmask['map']

    return imgmask, mask


def mask_rectangles(img, rectangles):
    """mask ``rectangles`` in ``img``

    :param img: pattern dict
    :param rectangles: list of ``sf.Rectangle``'s

    :returns: masked image, mask
    """
    imgmask = deepcopy(img)

    mask = np.ones_like(imgmask['map'], dtype=np.uint8)

    for rectangle in rectangles:
        temp_mask = np.ones_like(imgmask['map'], dtype=np.uint8)
        corner = rectangle.corner
        width = rectangle.width
        height = rectangle.height
        c = np.array([corner[0], corner[0] + width, corner[0] + width, corner[0]])
        r = np.array([corner[1], corner[1], corner[1] + height, corner[1] + height])
        rr, cc = sidraw.polygon(r, c, shape=imgmask['map'].shape)
        temp_mask[rr, cc] = 0
        mask *= temp_mask

    imgmask['map'] = imgmask['map'] * mask

    return imgmask, mask


def mask_polygons(img, polygons):
    """mask ``polygons`` in ``img``

    :param img: pattern dict
    :param polygons: list of ``sf.Polygon``'s

    :returns: masked image, mask
    """

    imgmask = deepcopy(img)

    mask = np.ones_like(imgmask['map'], dtype=np.uint8)

    for polygon in polygons:
        temp_mask = np.ones_like(imgmask['map'], dtype=np.uint8)
        xy = polygon.vertices
        c = xy[:, 0]
        r = xy[:, 1]
        rr, cc = sidraw.polygon(r, c, shape=imgmask['map'].shape)
        temp_mask[rr, cc] = 0
        mask *= temp_mask

    imgmask['map'] = imgmask['map'] * mask

    return imgmask, mask


def mask_ring_sectors(img, sectors):
    """mask ``sectors`` in ``img``

    :param img: pattern dict
    :param sectors: list of ``sf.RingSector``'s

    :returns: masked image, mask, masks
    """

    imgmask = deepcopy(img)

    mask = np.ones_like(imgmask['map'], dtype=np.uint8)

    masks = []
    for sector in sectors:
        temp_mask = np.ones_like(imgmask['map'], dtype=np.uint8)
        xy = SelectRingSectors.compute_path(sector)
        c = xy[:, 0]
        r = xy[:, 1]
        rr, cc = sidraw.polygon(r, c, shape=imgmask['map'].shape)
        temp_mask[rr, cc] = 0
        mask *= temp_mask
        masks.append(temp_mask)


    imgmask['map'] = imgmask['map'] * mask

    return imgmask, mask, masks


def make_peak_mask(img, peak_sectors):
    """mask ``peak_sectors`` in ``img``

    ``peak_sectors`` is a stacked list of sectors for the peak and the reference.

    This function returns a stacked list of masks [peak_mask, reference_mask]

    :param img: pattern dict
    :param peak_sectors: stacked list of ``sf.RingSector``'s

    :returns: masks
    """
    masks = []
    for sectors in peak_sectors:
        mask0 = np.zeros_like(img['map'], dtype=np.uint8)
        xy = SelectRingSectors.compute_path(sectors[0])
        c = xy[:, 0]
        r = xy[:, 1]
        rr, cc = sidraw.polygon(r, c, shape=img['map'].shape)
        mask0[rr, cc] = 1

        mask1 = np.zeros_like(img['map'], dtype=np.uint8)
        xy = SelectRingSectors.compute_path(sectors[1])
        c = xy[:, 0]
        r = xy[:, 1]
        rr, cc = sidraw.polygon(r, c, shape=img['map'].shape)
        mask1[rr, cc] = 1

        masks.append([mask0, mask1])

    return masks


def midpnt(img):
    """ compute the midpoint pixel of an image

    The pixel coordinates in dimension (``dim``) are computed as

        :if ``dim`` is odd: ``(dim - 1)/2``
        :if ``dim`` is even: ``dim/2``

    :param img: pattern
    :return: midpoint pixel in image coordinates
    """
    shape = np.asarray(img['map'].shape, dtype=np.int)
    midp = np.zeros_like(shape)
    for i, item in enumerate(shape):
        midp[i] = item - 1 if item % 2 else item

    midp = midp[::-1]
    return midp // 2


def extend_image(img, center):
    """extend the pattern such that the midpoint is the center of the image

    the shape of the extended image is always odd

    :param img: pattern
    :param center: center
    :return: extended pattern
    """
    array_mid = np.asarray(list(reversed(center)), dtype=np.int)
    map_size = np.asarray(img['map'].shape, dtype=np.int)
    map_mid = midpnt(img)[::-1]
    delta_midp = map_mid - array_mid

    ext_y = np.abs(2*array_mid[0] + 1 - map_size[0])
    if delta_midp[0] > 0:
        new_map = np.vstack((np.zeros((ext_y, map_size[1])), img['map']))
    else:
        new_map = np.vstack((img['map'], np.zeros((ext_y, map_size[1]))))

    ext_x = np.abs(2*array_mid[1] + 1 - map_size[1])
    if delta_midp[1] > 0:
        new_map = np.hstack((np.zeros((new_map.shape[0], ext_x)), new_map))
    else:
        new_map = np.hstack((new_map, np.zeros((new_map.shape[0], ext_x))))

    new_img = deepcopy(img)
    new_img['map'] = new_map

    return new_img


def harmonize_image(img):
    """harmonize the image"""

    new_map = img['map']

    map_size = np.asarray(img['map'].shape, dtype=np.int)
    if not any(map_size % 2):
        raise ValueError('The shape of the pattern must be odd.')

    map_mid = midpnt(img)[::-1]

    lower_left = new_map[0:map_mid[0]+1, 0:map_mid[1]+1]

    lower_right = new_map[0:map_mid[0]+1, map_mid[1]:]
    lower_right = np.fliplr(lower_right)

    upper_left = new_map[map_mid[0]:, 0:map_mid[1]+1]
    upper_left = np.flipud(upper_left)

    upper_right = new_map[map_mid[0]:, map_mid[1]:]
    upper_right = np.flipud(np.fliplr(upper_right))

    all_sum = np.zeros_like(lower_left)
    count = np.zeros_like(lower_left)
    for i in [lower_left, lower_right, upper_left, upper_right]:
        all_sum += i
        count += i > 0

    count[count == 0] = 1
    final_map = all_sum / count

    # we have to crop the parts as the row and column containing
    # the midpoint would otherwise appear four times in the final pattern
    ll = final_map
    lr = np.fliplr(ll)[:, 1:]
    ul = np.flipud(ll)[1:, :]
    ur = np.flipud(np.fliplr(ll))[1:, 1:]

    l = np.hstack((ll, lr))
    u = np.hstack((ul, ur))

    f = np.vstack((l, u))

    new_img = deepcopy(img)
    new_img['map'] = f

    return new_img


def harmonize_image_2(img):
    """harmonize the pattern, assumeing 180° symmetry"""
    new_map = img['map']

    map_size = np.asarray(img['map'].shape, dtype=np.int)
    if not any(map_size % 2):
        raise ValueError('The shape of the pattern must be odd.')

    map_mid = midpnt(img)[::-1]

    lower = new_map[0:map_mid[0]+1, :]
    lower = np.flipud(np.fliplr(lower))
    upper = new_map[map_mid[0]:, :]

    all_sum = np.zeros_like(lower)
    count = np.zeros_like(lower)
    for i in [lower, upper]:
        all_sum += i
        count += i > 0

    count[count == 0] = 1
    final_map = all_sum / count

    # we have to crop the parts as the row containing
    # the midpoint would otherwise appear twice in the final pattern
    u = final_map[1:, :]
    l = np.flipud(np.fliplr(final_map))

    f = np.vstack((l, u))

    new_img = deepcopy(img)
    new_img['map'] = f

    return new_img


def harmony(img, center, angle=None):
    """Harmonize the pattern by exploiting symmetry


    If the shape of the pattern is not anymore odd after the rotation has been
    performed the pattern is padded with zeros such that its shape is odd.


    :param img:  pattern
    :param center:  center coordinates in pattern
    :param angle: rotate image by angle, in degrees in counter-clockwise direction
    :return: harmonized pattern
    """

    if angle is not None:
        ext_img = extend_image(img, center)
        new_center = midpnt(ext_img)

        ext_img['map'] = sitransform.rotate(ext_img['map'], angle,
                                            center=new_center, cval=0,
                                            resize=1, preserve_range=1)

        # after rotation the image shape may not be odd anymore so we append zeros
        if not ext_img['map'].shape[0] % 2:
            fill = np.zeros((1, ext_img['map'].shape[1]))
            ext_img['map'] = np.vstack((fill, ext_img['map']))

        if not ext_img['map'].shape[1] % 2:
            fill = np.zeros((ext_img['map'].shape[0], 1))
            ext_img['map'] = np.hstack((fill, ext_img['map']))

        harmonized =  harmonize_image(ext_img)
    else:
        harmonized =  harmonize_image(extend_image(img, center))

    harmonized['beam_position'] = midpnt(harmonized)
    return harmonized


def harmony2(img, center):
    """Harmonize the pattern by exploiting rotational symmetry by 180°

    :param img:  pattern
    :param center:  center coordinates in pattern
    :return: harmonized pattern
    """
    harmonized = harmonize_image_2(extend_image(img, center))
    harmonized['beam_position'] = midpnt(harmonized)
    return harmonized


def local_median_filter(img, region, size=3):
    """Apply a median filter on a part of the pattern

    :param img: pattern
    :param region: region to apply median filter on, a ``sf.Rectangle``
    :param size: size of the filter (see ``scipy.ndimage.filters.median_filter``)
    :return: locally filtered pattern
    """
    if abs(region.width) < size or abs(region.height) < size:
        raise ValueError('Size must be greater or equal to region height and width.')

    new_map = img['map']

    # reverse the selection if height/with is negative
    cols = [region.corner[1], region.corner[1] + region.height][::np.sign(region.height, casting='unsafe', dtype=int)]
    rows = [region.corner[0], region.corner[0] + region.width][::np.sign(region.width, casting='unsafe', dtype=int)]
    slices = np.s_[cols[0]:cols[1], rows[0]:rows[1]]

    new_map[slices] = median_filter(new_map[slices], size)

    new_img = deepcopy(img)
    new_img['map'] = new_map

    return new_img

# ----------------------------------------------------------------------------
# Interaction classes
# ----------------------------------------------------------------------------

class InteractBase:
    """Base class for interactions"""
    def __init__(self):
        self._help_fig = None
        self._help_text = ""

    def show_help(self):
        """show the help window"""
        lines = []
        for line in self._help_text:
            lines.append('{:>20}:  {:<40}'.format(*line))

        help_text = "\n".join(lines)

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 0.8])
        ax.set_axis_off()
        fig.suptitle('Help', fontsize=20, fontweight='bold')

        ax.set_title('Assignment of keys:')

        # table looks strange
        # tab = ax.table(cellText=text, loc='center', edges="open", cellLoc='left')
        # tab.set_fontsize(15)

        ax.text(0.5, 0.95, help_text,
                horizontalalignment='center',
                verticalalignment='top',
                multialignment='center',
                linespacing=1.5,
                fontproperties=FontProperties(family='monospace', size='large')
               )

        self._help_fig = fig
        fig.canvas.mpl_connect('close_event', lambda event: plt.close(event.canvas.figure))
        fig.show()

    @staticmethod
    def _block(fig):
        logger.debug("starting blocking event loop")
        fig.canvas.mpl_connect('close_event', handle_close)
        fig.canvas.start_event_loop(timeout=-1)

    def __del__(self):
        if self._help_fig is not None:
            # we explicitly close the helper window, otherwise matplotlib will keep it in memory
            logger.debug("closing help figure")
            plt.close(self._help_fig)


class SelectCircles(InteractBase):
    """Select circles in fig

    :param fig: matplotlib figure object

    .. note:: If you use ``sf.show`` the figure must be created using ``block=False``

    **Usage**

    :Draw circle: left mouse button to set center, set radius by clicking left button again
    :Modify circle:
        use +/- keys to increase/decrease radius by 1 px

        use arrow-keys to move center
    :Delete circle: backspace
    :Select circle: use "ctrl+[0..6]" to select one of the first 7 circles
    """
    def __init__(self, fig):
        super().__init__()

        self.i = -1
        self.x = 0.0
        self.y = 0.0

        self.radius = None

        # keep track of matplotlib.patches.Circle instances
        self._circles = []

        self.fig = fig

        ax = self.fig.get_axes()[0]
        ax.format_coord = lambda x, y: "Press ctrl+h for help!        x={:6.3f} y={:6.3f}".format(x, y)

        self._help_text = [['+', 'bigger'],
                           ['-', 'smaller'],
                           ['left', 'move left'],
                           ['right', 'move right'],
                           ['up', 'move up'],
                           ['down', 'move down'],
                           ['backspace', 'delete circle'],
                           ['ctrl+[0..6]', 'select n-th circle']]

        logger.debug("connecting events")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

        self._block(self.fig)

    @property
    def circles(self):
        """list of ``sf.Circle``'s
        """
        return [Circle(center=c.center, radius=c.get_radius()) for c in self._circles]

    def on_button_press(self, event):
        """Capture button press events to draw circles"""
        if event.button == 1:
            logger.debug("event.button == 1")

            if self.radius is None:
                logger.debug('self.radius is None')
                self.x = event.xdata
                self.y = event.ydata
                self._circles.append(plt.Circle((self.x, self.y), 0.1, color='black', fill=False))
                event.inaxes.add_patch(self._circles[-1])
                self.radius = 0.1
                self.fig.canvas.draw()
            else:
                self.radius = None

    def on_move(self, event):
        """Capture mouse motion and set the radius of the circle"""
        if self._circles and event.inaxes and self.radius:
            tmp_radius = np.sqrt((self.x - event.xdata)**2 + (self.y - event.ydata)**2)
            self._circles[-1].set_radius(tmp_radius)
            self.fig.canvas.draw()

    def on_key_press(self, event):
        """Capture key press events to modify circles"""
        if event.key == 'ctrl+h':
            self.show_help()

        # select already drawn circle
        elif event.key == 'ctrl+0':
            self.i = 0

        elif event.key == 'ctrl+1':
            self.i = 1

        elif event.key == 'ctrl+2':
            self.i = 2

        elif event.key == 'ctrl+3':
            self.i = 3

        elif event.key == 'ctrl+4':
            self.i = 4

        elif event.key == 'ctrl+5':
            self.i = 5

        elif event.key == 'ctrl+6':
            self.i = 6

        elif self._circles:
            if event.key == '+':
                radius = self._circles[self.i].get_radius() + 1
                self._circles[self.i].set_radius(radius)

            elif event.key == '-':
                radius = self._circles[self.i].get_radius() - 1
                self._circles[self.i].set_radius(radius)

            elif event.key == 'right':
                self.x, self.y = self._circles[self.i].center
                self.x = self._circles[self.i].center[0] + 1
                self._circles[self.i].center = self.x, self.y

            elif event.key == 'left':
                self.x, self.y = self._circles[self.i].center
                self.x = self._circles[self.i].center[0] - 1
                self._circles[self.i].center = self.x, self.y

            elif event.key == 'up':
                self.x, self.y = self._circles[self.i].center
                self.y = self._circles[self.i].center[1] - 1
                self._circles[self.i].center = self.x, self.y

            elif event.key == 'down':
                self.x, self.y = self._circles[self.i].center
                self.y = self._circles[self.i].center[1] + 1
                self._circles[self.i].center = self.x, self.y

            elif event.key == 'backspace':
                logger.debug("removing circle: %d" % self.i)
                circle = self._circles.pop(self.i)
                circle.remove()

        self.fig.canvas.draw()


class SelectCirclesSameCenter(InteractBase):
    """Select circles in fig all have the same center

    :param fig: matplotlib figure object
    :param center: center position

    .. note:: If you use ``sf.show`` the figure must be created using ``block=False``

    **Usage**

    :Draw circle: left mouse button to set center, set radius by clicking left button again
    :Modify circle:
        use +/- keys to increase/decrease radius by 1 px

        use arrow-keys to move center
    :Delete circle: backspace
    :Select circle: use "ctrl+[0..6]" to select one of the first 7 circles
    """
    def __init__(self, fig, center):
        super().__init__()

        self.i = -1
        self.x = 0.0
        self.y = 0.0

        self.radius = None
        self.center = center
        self._center_mark = None

        # keep track of matplotlib.patches.Circle instances
        self._circles = []

        self.fig = fig

        ax = self.fig.get_axes()[0]
        ax.format_coord = lambda x, y: "Press ctrl+h for help!        x={:6.3f} y={:6.3f}".format(x, y)

        self._help_text = [['+', 'bigger'],
                           ['-', 'smaller'],
                           ['left', 'move left'],
                           ['right', 'move right'],
                           ['up', 'move up'],
                           ['down', 'move down'],
                           ['backspace', 'delete circle']]

        logger.debug("connecting events")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

        self._block(self.fig)

    @property
    def circles(self):
        """list of ``sf.Circle``'s
        """
        return [Circle(center=c.center, radius=c.get_radius()) for c in self._circles]

    def on_button_press(self, event):
        """Capture button press events to draw circles"""
        if event.button == 1:
            logger.debug("event.button == 1")

            if self.radius is None and self.center is None:
                logger.debug('self.radius is None and self.center is None')
                self.x = event.xdata
                self.y = event.ydata
                self.center = (self.x, self.y)
                self._center_mark = plt.Line2D([self.x], [self.y], ls='None', marker='o', c='r')
                self._circles.append(plt.Circle((self.x, self.y), 0.1, color='black', fill=False))
                event.inaxes.add_patch(self._circles[-1])
                event.inaxes.add_line(self._center_mark)
                self.radius = 0.1
                self.fig.canvas.draw()

            ## user has to click somewhere to start new circle
            elif self.radius is None and self.center is not None:
            #elif self.center is not None:
                self.x = self.center[0]
                self.y = self.center[1]
                logger.debug('self.radius is None and self.center is not None')
                if self._center_mark is None:
                    self._center_mark = plt.Line2D([self.center[0]], [self.center[1]], ls='None', marker='o', c='r')
                    event.inaxes.add_line(self._center_mark)
                self._circles.append(plt.Circle((self.x, self.y), 0.1, color='black', fill=False))
                event.inaxes.add_patch(self._circles[-1])
                self.radius = 0.1
                self.fig.canvas.draw()
            else:
                self.radius = None

    def on_move(self, event):
        """Capture mouse motion and set the radius of the circle"""
        if self._circles and event.inaxes and self.radius:
            tmp_radius = np.sqrt((self.x - event.xdata)**2 + (self.y - event.ydata)**2)
            self._circles[-1].set_radius(tmp_radius)
            self.fig.canvas.draw()

    def on_key_press(self, event):
        """Capture key press events to modify circles"""
        if event.key == 'ctrl+h':
            self.show_help()

        elif self._circles:
            if event.key == '+':
                radius = self._circles[self.i].get_radius() + 1
                self._circles[self.i].set_radius(radius)

            elif event.key == '-':
                radius = self._circles[self.i].get_radius() - 1
                self._circles[self.i].set_radius(radius)

            elif event.key == 'right':
                self.x += 1
            elif event.key == 'left':
                self.x -= 1
            elif event.key == 'up':
                self.y -= 1
            elif event.key == 'down':
                self.y += 1
            elif event.key == 'backspace':
                logger.debug("removing circle: %d" % self.i)
                circle = self._circles.pop(self.i)
                circle.remove()

            self.center = (self.x, self.y)
            for c in self._circles:
                c.center = self.x, self.y
            self._center_mark.set_data([self.center[0]], [self.center[1]])
        self.fig.canvas.draw()


class SelectRingSectors(InteractBase):
    """Select ring segments

    :param fig: matplotlib figure object
    :param center: (optional) center of the segment

    **Usage**

    :Draw sector: left mouse button to set vertices, adjust position with keyboard (see ctrl-h), press space to draw new sector

    :Delete sector: backspace

    """
    def __init__(self, fig, center=None):
        super().__init__()

        self.center = np.asarray(center) if center is not None else None

        # keep track of matplotlib.patches.Wedge instances
        self._wedges = []

        self.fig = fig

        self.a = None
        self.b = None
        self.l = None
        self._wedge_done = False

        ax = self.fig.get_axes()[0]
        ax.format_coord = lambda x, y: "Press ctrl+h for help!        x={:6.3f} y={:6.3f}".format(x, y)

        self._help_text = [['left', 'move 1st point left'],
                           ['right', 'move 1st point right'],
                           ['up', 'move 1st point up'],
                           ['down', 'move 1st point down'],
                           ['shift+left', 'move 2nd point left'],
                           ['shift+right', 'move 2nd point right'],
                           ['shift+up', 'move 2nd point up'],
                           ['shift+down', 'move 2nd point down'],
                           ['backspace', 'delete sector'],
                           ['space', 'finish current sector']]

        logger.debug("connecting events")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

        # ax.add_patch(Wedge((600, 200), 50, 0, 60, width=20,
        #                           color='black', fill=False))
        # self.fig.canvas.draw()

        self._block(self.fig)

    @property
    def sectors(self):
        """list of ``sf.RingSectors``'s
        """
        return [RingSector(i.theta1, i.theta2, i.r, i.width, i.center) for i in self._wedges]

    def _update_wedge(self):
        ac = self.a - self.center
        bc = self.b - self.center
        r1 = np.sqrt(np.sum(ac ** 2))
        r2 = np.sqrt(np.sum(bc ** 2))
        alpha = np.arctan2(ac[1], ac[0]) * 180 / np.pi
        beta = np.arctan2(bc[1], bc[0]) * 180 / np.pi

        w = self._wedges[-1]
        w.set_radius(r2)
        w.set_width(r2 - r1)
        w.set_theta1(alpha)
        w.set_theta2(beta)

        logger.debug('a = [%.2f, %.2f], b = [%.2f, %2f], alpha = %.1f, beta = %.1f' \
                     %(self.a[0], self.a[1], self.b[0], self.b[1], alpha, beta))

        self.fig.canvas.draw()

    def on_button_press(self, event):
        """Capture button press events to draw circles"""
        if event.button == 1 and not self._wedge_done:
            if self.center is None:
                logger.debug('selecting center')
                self.center = np.array([event.xdata, event.ydata])

            elif self.a is None and self.center is not None:
                self.l.remove()
                self.a = np.array([event.xdata, event.ydata])
                self.b = self.a + np.array([0.1, 0.1])

                ac = self.a - self.center
                bc = self.b - self.center
                r1 = np.sqrt(np.sum(ac**2))
                r2 = np.sqrt(np.sum(bc**2))
                alpha = np.arctan2(ac[1], ac[0])* 180 / np.pi
                beta = np.arctan2(bc[1], bc[0])* 180 / np.pi

                self._wedges.append(Wedge(self.center, r2, alpha, beta,
                                          width=r2-r1,
                                          color='black',
                                          fill=False))
                ax = event.inaxes
                ax.add_patch(self._wedges[-1])
                self.fig.canvas.draw()

            elif self.a is not None and not self._wedge_done:
                self.b = np.array([event.xdata, event.ydata])
                self._update_wedge()
                self._wedge_done = True

    def on_move(self, event):
        """Capture mouse motion and set the radius of the circle"""

        if self.center is not None and self.l is None and event.inaxes:
            self.l = plt.Line2D((self.center[0], event.xdata), (self.center[1], event.ydata), ls='--', c='k')
            ax = event.inaxes
            ax.add_line(self.l)
            self.fig.canvas.draw()
        elif self.center is not None:
            self.l.set_data((self.center[0], event.xdata), (self.center[1], event.ydata))
            self.fig.canvas.draw()

        if not self._wedge_done and self._wedges and self.a is not None and event.inaxes:
            self.b = np.array([event.xdata, event.ydata])
            self._update_wedge()

    def on_key_press(self, event):
        """Capture key press events to modify circles"""
        if event.key == 'ctrl+h':
            self.show_help()
        if self._wedges:
            if event.key == ' ':  # space
                self.a = None
                self.b = None
                self.l = None
                self._wedge_done = False

            elif event.key == 'up':
                self.a[1] += 1
                self._update_wedge()
            elif event.key == 'down':
                self.a[1] -= 1
                self._update_wedge()
            elif event.key == 'left':
                self.a[0] -= 1
                self._update_wedge()
            elif event.key == 'right':
                self.a[0] += 1
                self._update_wedge()
            elif event.key == 'shift+up':
                self.b[1] += 1
                self._update_wedge()
            elif event.key == 'shift+down':
                self.b[1] -= 1
                self._update_wedge()
            elif event.key == 'shift+left':
                self.b[0] -= 1
                self._update_wedge()
            elif event.key == 'shift+right':
                self.b[0] += 1
                self._update_wedge()
            elif event.key == 'backspace':
                logger.debug("removing wedge")
                w = self._wedges.pop()
                w.remove()
                self.a = None
                self.b = None
                self.l = None
                self._wedge_done = False

        self.fig.canvas.draw()

    @classmethod
    def compute_path(cls, RingSector, n=1000):
        """create a polygon of a RingSector

        :param RingSector: a ``sf.RingSector``
        :param n: number of sampling point for arc
        :return: vertices
        """
        # code adapted from matplotlib/patches.py
        theta1 = RingSector.theta1
        theta2 = RingSector.theta2
        r = RingSector.radius
        width = RingSector.width
        center = RingSector.center

        if abs((theta2 - theta1) - 360) <= 1e-12:
            theta1, theta2 = 0, 360
            connector = Path.MOVETO
        else:
            connector = Path.LINETO

            # Form the outer ring
        arc = Path.arc(theta1, theta2, n=n)

        # Partial annulus needs to draw the outer ring
        # followed by a reversed and scaled inner ring
        v1 = arc.vertices
        v2 = arc.vertices[::-1] * float(r - width) / r
        v = np.vstack([v1, v2, v1[0, :], (0, 0)])
        c = np.hstack([arc.codes, arc.codes, connector, Path.CLOSEPOLY])
        c[len(arc.codes)] = connector

        # Shift and scale the wedge to the final location.
        v *= r
        v += np.asarray(center)
        p = Path(v, c)

        return p.vertices


class SelectTwoRingSectors(InteractBase):
    """Select two ring segments

    the second segment has only a rotated starting point

    :param fig: matplotlib figure object
    :param center: (optional) center of the segment

    **Usage**

    :Draw sector: left mouse button to set vertices, adjust position with keyboard (see ctrl-h), press space to draw new sector

    :Delete sector: backspace

    """
    def __init__(self, fig, center=None):
        super().__init__()

        if center is None:
            raise NotImplementedError('sorry, you have to provide a center')

        self.center = np.asarray(center) if center is not None else None
        self._center_mark = plt.Line2D([self.center[0]], [self.center[1]], ls='None', marker='o', c='r')

        # keep track of matplotlib.patches.Wedge instances
        self._wedges = []

        self.fig = fig

        self.a = None
        self.b = None
        self.theta21 = 0
        self.l = None
        self._wedge_done = False
        self._2nd_wedge_done = False

        ax = self.fig.get_axes()[0]
        ax.add_line(self._center_mark)
        ax.format_coord = lambda x, y: "Press ctrl+h for help!        x={:6.3f} y={:6.3f}".format(x, y)

        self._help_text = [['left', 'move 1st point left'],
                           ['right', 'move 1st point right'],
                           ['up', 'move 1st point up'],
                           ['down', 'move 1st point down'],
                           ['shift+left', 'move 2nd point left'],
                           ['shift+right', 'move 2nd point right'],
                           ['shift+up', 'move 2nd point up'],
                           ['shift+down', 'move 2nd point down'],
                           ['+', 'increase angle for 2nd sector'],
                           ['-', 'decrease angle for 2nd sector'],
                           ['backspace', 'delete current sectors'],
                           ['space', 'finish current sectors']]

        logger.debug("connecting events")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

        # ax.add_patch(Wedge((600, 200), 50, 0, 60, width=20,
        #                           color='black', fill=False))
        # self.fig.canvas.draw()

        self._block(self.fig)

    @property
    def sectors(self):
        """stacked list of ``sf.RingSectors``'s
        """
        return [[RingSector(i[0].theta1, i[0].theta2, i[0].r, i[0].width, i[0].center),
                 RingSector(i[1].theta1, i[1].theta2, i[1].r, i[1].width, i[1].center)] for i in self._wedges]

    def _update_wedge(self):
        ac = self.a - self.center
        bc = self.b - self.center
        r1 = np.sqrt(np.sum(ac ** 2))
        r2 = np.sqrt(np.sum(bc ** 2))
        alpha = np.arctan2(ac[1], ac[0]) * 180 / np.pi
        beta = np.arctan2(bc[1], bc[0]) * 180 / np.pi

        w = self._wedges[-1][0]
        w.set_radius(r2)
        w.set_width(r2 - r1)
        w.set_theta1(alpha)
        w.set_theta2(beta)

        w2 = self._wedges[-1][1]
        if w2:
            w2.set_radius(r2)
            w2.set_width(r2 - r1)
            w2.set_theta1(self.theta21)
            w2.set_theta2(self.theta21 + (beta-alpha))

        self.fig.canvas.draw()

    def on_button_press(self, event):
        """Capture button press events to draw circles"""
        if event.button == 1:
            if not self._wedge_done:
                if self.center is None:
                    logger.debug('selecting center')
                    self.center = np.array([event.xdata, event.ydata])

                elif self.a is None and self.center is not None:
                    self.l.remove()
                    self.a = np.array([event.xdata, event.ydata])
                    self.b = self.a + np.array([0.1, 0.1])

                    ac = self.a - self.center
                    bc = self.b - self.center
                    r1 = np.sqrt(np.sum(ac**2))
                    r2 = np.sqrt(np.sum(bc**2))
                    alpha = np.arctan2(ac[1], ac[0])* 180 / np.pi
                    beta = np.arctan2(bc[1], bc[0])* 180 / np.pi

                    self._wedges.append([Wedge(self.center, r2, alpha, beta,
                                               width=r2-r1, color='black',
                                               fill=False), None])
                    ax = event.inaxes
                    ax.add_patch(self._wedges[-1][0])
                    self.fig.canvas.draw()

                elif self.a is not None:
                    self.b = np.array([event.xdata, event.ydata])
                    self._update_wedge()
                    self._wedge_done = True

            elif self._wedge_done and not self._2nd_wedge_done:
                self._2nd_wedge_done = True

    def on_move(self, event):
        """Capture mouse motion and set the radius of the circle"""
        ax = event.inaxes
        if self.center is not None and self.l is None and event.inaxes:
            self.l = plt.Line2D((self.center[0], event.xdata), (self.center[1], event.ydata), ls='--', c='k')
            ax.add_line(self.l)

        elif self.center is not None:
            self.l.set_data((self.center[0], event.xdata), (self.center[1], event.ydata))

        if not self._wedge_done and self._wedges and self.a is not None and event.inaxes:
            self.b = np.array([event.xdata, event.ydata])
            self._update_wedge()

        if self._wedge_done and not self._2nd_wedge_done and event.inaxes:
            ww = self._wedges[-1]

            ac = np.array([event.xdata, event.ydata]) - self.center
            alpha = np.arctan2(ac[1], ac[0]) * 180 / np.pi
            beta = alpha+(ww[0].theta2-ww[0].theta1)
            self.theta21 = alpha
            if ww[1]:
                w = ww[1]
                w.set_theta1(alpha)
                w.set_theta2(beta)
            else:
                logger.debug('making second sector')
                w = Wedge(ww[0].center, ww[0].r, alpha, beta,
                          width=ww[0].width, color='black', fill=False)
                ax.add_patch(w)
                self._wedges[-1][1] = w
        self.fig.canvas.draw()

    def on_key_press(self, event):
        """Capture key press events to modify circles"""
        if event.key == 'ctrl+h':
            self.show_help()
        if self._wedges:
            if event.key == ' ':  # space
                self.a = None
                self.b = None
                self.l = None
                self._wedge_done = False
                self._2nd_wedge_done = False
                self.theta21 = 0

            elif event.key == 'up':
                self.a[1] += 1
            elif event.key == 'down':
                self.a[1] -= 1
            elif event.key == 'left':
                self.a[0] -= 1
            elif event.key == 'right':
                self.a[0] += 1
            elif event.key == 'shift+up':
                self.b[1] += 1
            elif event.key == 'shift+down':
                self.b[1] -= 1
            elif event.key == 'shift+left':
                self.b[0] -= 1
            elif event.key == 'shift+right':
                self.b[0] += 1
            elif event.key == '+':
                self.theta21 += 0.5
            elif event.key == '-':
                self.theta21 -= 0.5
            elif event.key == 'backspace':
                logger.debug("removing wedge")
                w = self._wedges.pop()
                w[0].remove()
                w[1].remove()
                self.a = None
                self.b = None
                self.l = None
                self._wedge_done = False
                self._2nd_wedge_done = False
                self.theta21 = 0

            if self._2nd_wedge_done:
                self._update_wedge()
        self.fig.canvas.draw()


class SelectRectangles(InteractBase):
    """Select rectangles in fig

    :param fig: matplotlib figure object

    .. note:: If you use ``sf.show`` the figure must be created using ``block=False``

    **Usage**

    :Draw rectangle: left mouse button to set corner, set other corner by clicking left button again
    :Modify circle:
        use +/-/*/_  keys to increase/decrease x/y by 1 px

        use arrow-keys to first corner

    :Delete rectangle: backspace
    """
    def __init__(self, fig):
        super().__init__()

        self.x = 0.0
        self.y = 0.0

        self.size = None  # [width, height]

        self._rectangles = []

        self.fig = fig

        ax = self.fig.get_axes()[0]
        ax.format_coord = lambda x, y: "Press ctrl+h for help!        x={:6.3f} y={:6.3f}".format(x, y)

        self._help_text = [['+', 'bigger in x-direction'],
                           ['-', 'smaller in x-direction'],
                           ['*', 'bigger in y-direction'],
                           ['_', 'smaller in y-direction'],
                           ['left', 'move left'],
                           ['right', 'move right'],
                           ['up', 'move up'],
                           ['down', 'move down'],
                           ['backspace', 'delete rectangle']]

        logger.debug("connecting events")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

        self._block(self.fig)

    @property
    def rectangles(self):
        """list of ``sf.Rectangle``'s
        """
        return [Rectangle(corner=r.get_xy(),
                          width=r.get_width(),
                          height=r.get_height()) for r in self._rectangles]

    def on_button_press(self, event):
        """Capture button press events to start drawing a rectangle"""
        if event.button == 1:
            logger.debug("event.button == 1")
            if self.size is None:
                logger.debug('self.size is None')
                self.x = event.xdata
                self.y = event.ydata
                self._rectangles.append(plt.Rectangle((self.x, self.y), 0.1, 0.1,
                                                      color='black', fill=False))
                ax = event.inaxes
                ax.add_patch(self._rectangles[-1])
                self.size = [0.1, 0.1]
                self.fig.canvas.draw()

            else:
                self.size = None

    def on_move(self, event):
        """Capture mouse motion and set the width and height of the rectangle"""
        if self._rectangles and event.inaxes and self.size:
            self._rectangles[-1].set_height(event.ydata - self.y)
            self._rectangles[-1].set_width(event.xdata - self.x)
            self.fig.canvas.draw()

    def on_key_press(self, event):
        """Capture key press events to modify rectangles"""
        if event.key == '+':
            self._rectangles[-1].set_width(self._rectangles[-1].get_width() + 1)

        elif event.key == '-':
            self._rectangles[-1].set_width(self._rectangles[-1].get_width() - 1)

        elif event.key == '*':
            self._rectangles[-1].set_height(self._rectangles[-1].get_height() + 1)

        elif event.key == '_':
            self._rectangles[-1].set_height(self._rectangles[-1].get_height() - 1)

        elif event.key == 'right':
            self.x += 1
            self._rectangles[-1].xy = self.x, self.y

        elif event.key == 'left':
            self.x -= 1
            self._rectangles[-1].xy = self.x, self.y

        elif event.key == 'up':
            self.y -= 1
            self._rectangles[-1].xy = self.x, self.y

        elif event.key == 'down':
            self.y += 1
            self._rectangles[-1].xy = self.x, self.y

        elif event.key == 'backspace':
            try:
                rect = self._rectangles.pop()
                rect.remove()
            except IndexError:
                pass

        elif event.key == 'ctrl+h':
            self.show_help()

        self.fig.canvas.draw()


class SelectPolygons(InteractBase):
    """Select polygons in fig

    :param fig: matplotlib figure object

    .. note:: If you use ``sf.show`` the figure must be created using ``block=False``

    **Usage**

    :Draw polygon: left mouse button to set vertices, set last vertex with right mouse button
    :Modify polygon:
        use `shift-backspace` to delete a vertex

        use arrow keys to move last vertex

    :Delete polygon: backspace
    """
    def __init__(self, fig):
        super().__init__()

        self.x = None
        self.y = None

        self._vertices = []
        self._polygons = []
        self._is_new_polygon = True

        self.fig = fig

        ax = self.fig.get_axes()[0]
        ax.format_coord = lambda x, y: "Press ctrl+h for help!        x={:6.3f} y={:6.3f}".format(x, y)

        self._help_text = [['left', 'move last vertex left'],
                           ['right', 'move last vertex right'],
                           ['up', 'move last vertex up'],
                           ['down', 'move last vertex down'],
                           ['backspace', 'delete whole polygon'],
                           ['shift+backspace', 'delete last vertex'],
                           ['right mouse click', 'complete polygon']]

        logger.debug("connecting events")
        self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

        self._block(self.fig)

    @property
    def polygons(self):
        """list of ``sf.Polygon``'s
        """
        return [Polygon(vertices=p.get_xy()) for p in self._polygons]

    def on_button_press(self, event):
        """Capture button press events to draw polygons"""
        x, y = event.xdata, event.ydata
        if event.button == 1:
            if self._vertices:
                self._vertices.append((x, y))
                self._polygons[-1].xy = self._vertices
            else:
                self._vertices.append((x, y))
                self._polygons.append(plt.Polygon(self._vertices, True, color='black', fill=False))
                ax = event.inaxes
                ax.add_patch(self._polygons[-1])

        elif event.button == 3 and self._vertices:
            self._vertices.append((x, y))
            self._polygons[-1].set_xy(self._vertices)
            self._vertices = []

        self.fig.canvas.draw()

    def on_move(self, event):
        """Capture mouse motion and update current vertex position"""
        if self._polygons and event.inaxes and self._vertices:
            self._polygons[-1].set_xy(self._vertices + [(event.xdata, event.ydata)])
            self.fig.canvas.draw()

    def on_key_press(self, event):
        """Capture key press events to modify polygons"""
        if event.key == 'shift+backspace':
            try:
                self._vertices.pop()
                if self._vertices:
                    self._polygons[-1].set_xy(self._vertices)
                else:
                    poly = self._polygons.pop()
                    poly.remove()
            except IndexError:
                pass

        elif event.key == 'backspace':
            try:
                poly = self._polygons.pop()
                poly.remove()
                self._vertices = []
            except IndexError:
                pass

        elif event.key == 'ctrl+h':
            self.show_help()

        elif event.key == 'right':
            if self._vertices:
                self._vertices[-1] = (self._vertices[-1][0] + 1, self._vertices[-1][1])
                self._polygons[-1].set_xy(self._vertices)

        elif event.key == 'left':
            if self._vertices:
                self._vertices[-1] = (self._vertices[-1][0] - 1, self._vertices[-1][1])
                self._polygons[-1].set_xy(self._vertices)

        elif event.key == 'up':
            if self._vertices:
                self._vertices[-1] = (self._vertices[-1][0], self._vertices[-1][1] - 1)
                self._polygons[-1].set_xy(self._vertices)

        elif event.key == 'down':
            if self._vertices:
                self._vertices[-1] = (self._vertices[-1][0], self._vertices[-1][1] + 1)
                self._polygons[-1].set_xy(self._vertices)

        self.fig.canvas.draw()


class RotatePattern(InteractBase):
    """Rotate patter

    :param fig: matplotlib figure object; the figure must be created using ``sf.show``
    :param img: pattern dict

    .. note:: If you use ``sf.show`` the figure must be created using ``block=False``

    **Usage**

    :rotate clockwise:  ``r``: 0.3°  ``R``: 1°
    :rotate anticlockwise:  ``a``: 0.3°  ``A``: 1°
    """
    def __init__(self, fig, img):
        super().__init__()

        self._img = img
        self._rotated_img = deepcopy(self._img)
        self._angle = 0

        self.fig = fig

        if not hasattr(self.fig, "_sf_kwargs_for_prepare"):
            raise TypeError("fig was not created using sf.show")

        self._ax = self.fig.get_axes()[0]
        self._image_axes = self._ax.get_images()[0]

        self.ax = self.fig.get_axes()[0]
        self.ax.format_coord = lambda x, y: "Press ctrl+h for help!        x={:6.3f} y={:6.3f}".format(x, y)

        self._help_text = [['r', 'rotate clockwise 0.3°'],
                           ['R', 'rotate clockwise 1°'],
                           ['a', 'rotate anticlockwise 0.3°'],
                           ['A', 'rotate anticlockwise 1°']]

        logger.debug("connecting events")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('close_event', self.on_close)

        self._block(self.fig)

    @property
    def img(self):
        """
        :return: rotated pattern
        :rtype: pattern dict
        """
        return self._img

    @property
    def angle(self):
        """
        :return: rotation angle
        :rtype: float
        """
        return self._angle

    def on_close(self, event):
        """set img to the rotated img when figure is closed"""
        self._img = self._rotated_img

    def _rotate(self, angle):
        """rotate the image and show it"""
        self._angle += angle
        self._rotated_img['map'] = ndimage.rotate(self._img['map'], self._angle, mode='constant', cval=0.0)
        self._rotated_img['beam_position'] = midpnt(self._rotated_img)
        img, ulev, dlev = prepare_patter_for_show(self._rotated_img, **self.fig._sf_kwargs_for_prepare)
        self._image_axes.set_data(img['map'])
        self._image_axes.set_clim(vmin=dlev, vmax=ulev)
        w, h = figaspect(img['map'])
        self.fig.set_size_inches(1.2*w, 1.2*h, forward=True)
        self.ax.set_aspect('auto')
        self.fig.canvas.draw()

    def on_key_press(self, event):
        """handle key press events"""
        if event.key == 'r':
            self._rotate(0.3)
        elif event.key == 'R':
            self._rotate(1)
        elif event.key == 'a':
            self._rotate(-0.3)
        elif event.key == 'A':
            self._rotate(-1)
        elif event.key == 'ctrl+h':
            self.show_help()

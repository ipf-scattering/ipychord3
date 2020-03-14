# Copyright (c) 2020 ipychord3 authors

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

"""utility functions for scattering analysis

"""

import re
from glob import glob

import numpy as np

from skimage.draw import circle
from scipy import sparse
from scipy.sparse.linalg import spsolve

####---------------------------------------------------------------------------
## Coordinate transforms

def to_polar(x, y):
    """convert ``x,y`` to ``r,phi``"""
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return r, phi


def to_cartesian(r, phi):
    """convert ``r,phi`` to ``x,y``"""
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return x, y

####---------------------------------------------------------------------------
## Baseline fitting

def baseline_als(y, lam, p, niter=10):
    """Baseline fitting using asymmetric least squares smoothing

    see: Eilers, Paul HC, and Hans FM Boelens. "Baseline correction with asymmetric least squares smoothing." Leiden University Medical Centre Report 1 (2005)

    :param y: signal, it is assumed to be evenly spaced
    :param lam: "smoothness", 10**2 <= lam <= 10**9
    :param p: "asymmetry",  0.001 <= p <= 0.1
    :param niter: number of iterations

    :return: the baseline
    """
    m = len(y) #y.shape[0]

    D = sparse.diags([1, -2, 1], [-2, -1, 0], shape=(m, m - 2), format='csr')
    # = np.diff(np.eye(m), 2)
    # D is the transpose of the one in the paper

    w = np.ones(m)
    z = np.zeros_like(y)

    for _ in range(niter):
        W = sparse.spdiags(w, 0, m, m)
        z[:] = sparse.linalg.spsolve(W + lam * D.dot(D.transpose()), w*y)
        w[:] = p * (y > z) + (1-p) * (y < z)
    return z

####---------------------------------------------------------------------------
## Path sorting

def sortedScanPaths(path_glob):
    """Given a globe for an online scan return the properly sorted paths.

    The files will be sorted like:

        ``['scan01_r1.ext', 'scan01_r2.ext', ..., 'scan01_r10.ext', ..., 'scan01_r20.ext', ...]``
    
    :param path_glob: glob for path

    :return: sorted paths
    """
    return sorted(glob(path_glob), key=lambda x: int(re.match(r'.*r(\d+)\..*', x).group(1)))

####---------------------------------------------------------------------------
## Data selection and modification

def indices_below(a, val, extend_window: int=0, extend_dir='-'):
    """Get array of bools where ``a<val`` and extend the indices 
    by ``extend_window``.
    
    :param a: 1d array
    :param val: the value below which the data should be
    :param extend_window: extend the window by n-points symmetrically
    :type extend_window: int
    :param extend_dir: 
        gives the direction of the extension

        * ``+``: extend to larger values 
        * ``-``: extend towards smaller values

    :return: array of bools
    """
    if extend_dir == '-':
        inv = True
    elif extend_dir == '+':
        inv = False
    else:
        raise ValueError("'extend_dir' can only have '+' or '-' as value")
    nz = np.nonzero(a<val)[0]
    idx = np.zeros_like(a, dtype=np.bool)
    idx[_extend_window(nz, extend_window, a.shape[0], inv)] = 1
    return idx


def indices_above(a, val, extend_window: int=0, extend_dir='+'):
    """Get array of bools where ``a>=val`` and extend the indices 
    by ``extend_window``.
    
    :param a: 1d array
    :param val: the value above which the data should be
    :param extend_window: extend the window by n-points symmetrically
    :type extend_window: int
    :param extend_dir: 
        gives the direction of the extension

        * ``+``: extend to larger values 
        * ``-``: extend towards smaller values
    
    :return: array of bools
    """
    idx_ = indices_below(a, val, extend_window, extend_dir)
    idx = np.ones_like(a, dtype=np.bool)
    idx[idx_] = 0
    return idx


def remove_around(a, index: int, extend: int):
    """Get array of bools like ``a`` where indices around ``index``
    are ``False`` and extended by ``extend``
    
    :param a: 1d array
    :param index: the index around which to remove the data
    :type index: int
    :param extend: extend the window by n-points symmetrically
    :type extend: int

    :return: array of bools
    """
    idx = np.ones_like(a, dtype=np.bool)
    idx[_extend_window(np.array([index,]), extend, a.shape[0], inv=False)] = 0
    return idx


def invert_indices(indices, length):
    mask = np.ones(length, np.bool)
    mask[indices] = 0
    return np.arange(length, dtype=np.int64)[mask]


def _extend_window(win, extend, length, inv=True):
    if inv:
        win = invert_indices(win, length)
    windowed = []
    for i in range(1,extend+1): # can probaly be done more elegantly
        windowed.append(win - i)
        windowed.append(win + i)
    
    windowed.append(win)
    idxs = np.unique(windowed)

    idxs = idxs[idxs>=0]
    idxs = idxs[idxs<length]
    if inv:
        idxs = invert_indices(idxs, length)
    return idxs

####---------------------------------------------------------------------------
## Test data

def gaussian2d(x, y, A, sigma, x0):
    """2d Gaussian in cartesian coordinates

    :param x: x-coordinates
    :param y: y-coordinates
    :param A: peak intensity
    :param sigma: stdev (stdev_x, stdev_y)
    :param x0: peak position (x_0, y_0)

    :return: 2d Gaussian
    """
    Z = A * np.exp(-( (x-x0[0])**2/(2*sigma[0]**2) + (y-x0[1])**2/(2*sigma[1]**2)))
    return Z


def gaussian2d_polar(x, y, A, sigma, x0):
    """2d Gaussian in polar coordinates

    :param x: x-coordinates (Cartesian)
    :param y: y-coordinates (Cartesian)
    :param A: peak intensity
    :param sigma: stdev (stdev_x radius, stdev_y angle (rad))
    :param x0: peak position (x_0 radius, y_0 angle (rad))

    :return: 2d Gaussian
    """
    R, PHI = to_polar(x, y)
    Z = A * np.exp(-( (R-x0[0])**2/(2*sigma[0]**2) + (PHI-x0[1])**2/(2*sigma[1]**2)))
    return Z


class TestPattern:
    """Create a test pattern

    The default values mimic a Pilatus 300k pattern with some reflexes.

    :param shape: shape of the pattern
    :param beam_position: beam position

    Usage:

    >>> p = TestPattern().make_pattern(-12)
    >>>
    """
    def __init__(self, shape=(619, 487), beam_position=(350, 300)):
        self._shape = shape
        self._beam_position = beam_position
        
        self.map = np.zeros(self._shape, dtype=np.int32)
        
        x = np.arange(self._shape[1], dtype=np.float)
        y = np.arange(self._shape[0], dtype=np.float)      
        self._X, self._Y = np.meshgrid(x, y)
        self._X_centered, self._Y_centered = np.meshgrid(x-self._beam_position[0], y-self._beam_position[1])
        
    def make_pattern(self, rot=0):
        """Return a rotated pattern

        Parameters:
        :param rot: rotation of the reflexes in degrees, counterclockwise is positive

        :return: the pattern dict
        """
        self.map = np.zeros(self._shape, dtype=np.int32)

        self.add_gaussian()
        self.add_reflex_1(A=500, sigma=(5, 6), x0=(220, rot))
        self.add_reflex_2(A=750, sigma=(6, 8), x0=(260, rot))
        self.add_beamstop()
        self.add_bars()
        
        p = dict()
        p['map'] = self.map
        p['beam_position'] = self._beam_position
        p['title'] = 'test pattern'
        
        return p
        
    def add_gaussian(self, A=1000, sigma=(200, 200), x0=None):
        """add a "background" Gaussian at ``x0``

        if ``x0=None`` use ``beam_position``
        """
        if x0 is None:
            x0 = self._beam_position
            
        self.map += np.asarray(np.round(
                        gaussian2d(self._X, self._Y, A, sigma, x0)), dtype=np.int32)
        
    def add_beamstop(self, size=40, pos=None):
        """add a beamstop
        
        :param size: size of beamstop
        :param pos: position, if not ``None`` overwrites ``beam_position``
        """
        
        if pos is None:
            pos = self._beam_position
        img_c = np.ones(self.map.shape, dtype=np.uint8)
        rr, cc = circle(*pos[::-1], size)
        img_c[rr, cc] = 0
        
        self.map *= img_c
    
    def add_bars(self, bars=[(195, 211), (407, 423)]):
        """add some bars"""
        for bar in bars:
            self.map[bar[0]:bar[1]+1,:] = 0
        
    def add_reflex_1(self, A=500, sigma=(5, 6), x0=(220, -20)):
        """``x_0 = (r0, phi0)``"""
        rot = x0[1]
        rot_rad = np.deg2rad(rot)
        r = x0[0]
        sigma = (sigma[0], np.deg2rad(sigma[0]))

        self.map += np.asarray(np.round(
                        gaussian2d_polar(self._X_centered, self._Y_centered,
                                         A, sigma, (r,  np.pi/2 - rot_rad))), dtype=np.int32)
        self.map += np.asarray(np.round(
                        gaussian2d_polar(self._X_centered, self._Y_centered,
                                         A, sigma, (r, -np.pi/2 - rot_rad))), dtype=np.int32)
        
    def add_reflex_2(self, A=750, sigma=(6, 8), x0=(260, -20)):
        """``x_0 = (r0, phi0)``"""
        rot = x0[1]
        rot_rad = np.deg2rad(rot)
        r = x0[0]
        sigma = (sigma[0], np.deg2rad(sigma[0]))
        
        angle = np.deg2rad(30)
        
        self.map += np.asarray(np.round(
                        gaussian2d_polar(self._X_centered, self._Y_centered,
                                         A, sigma, (r,  (np.pi/2 + angle) - rot_rad))), dtype=np.int32)
        self.map += np.asarray(np.round(
                        gaussian2d_polar(self._X_centered, self._Y_centered,
                                         A, sigma, (r, -(np.pi/2 + angle) - rot_rad))), dtype=np.int32)
        self.map += np.asarray(np.round(
                        gaussian2d_polar(self._X_centered, self._Y_centered,
                                         A, sigma, (r,  (np.pi/2 - angle) - rot_rad))), dtype=np.int32)
        self.map += np.asarray(np.round(
                        gaussian2d_polar(self._X_centered, self._Y_centered,
                                         A, sigma, (r, -(np.pi/2 - angle) - rot_rad))), dtype=np.int32)

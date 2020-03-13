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

import pytest
from unittest.mock import Mock

import numpy as np
from numpy import testing

from ipychord3 import utils

class TestSortedScanPaths:
    def test_win_path(self):
        paths_desired = [r'C:\test\scanr1.ext',
                         r'C:\test\scanr2.ext',
                         r'C:\test\scanr3.ext',
                         r'C:\test\scanr10.ext',
                         r'C:\test\scanr11.ext',
                         r'C:\test\scanr20.ext' ]
        paths_in = [r'C:\test\scanr1.ext',
                    r'C:\test\scanr10.ext',
                    r'C:\test\scanr11.ext',
                    r'C:\test\scanr2.ext',
                    r'C:\test\scanr20.ext'
                    r'C:\test\scanr3.ext']

        # mock glob
        glob = Mock()
        glob.glob.return_value = paths_in

        out = utils.sortedScanPaths('test')

        assert all([a == b for a, b in zip(paths_desired, out)])

    def test_unix_path(self):
        paths_desired = ['/test/scanr1.ext',
                         '/test/scanr2.ext',
                         '/test/scanr3.ext',
                         '/test/scanr10.ext',
                         '/test/scanr11.ext',
                         '/test/scanr20.ext' ]
        paths_in = ['/test/scanr1.ext',
                    '/test/scanr10.ext',
                    '/test/scanr11.ext',
                    '/test/scanr2.ext',
                    '/test/scanr20.ext'
                    '/test/scanr3.ext']

        # mock glob
        glob = Mock()
        glob.glob.return_value = paths_in

        out = utils.sortedScanPaths('test')

        assert all([a == b for a, b in zip(paths_desired, out)])


class TestIndexAbove:
    def test_no_extend(self):
        a = np.array([1, 2, 3, 4, 3, 2, 1])
        desired = np.array([0, 0, 1, 1, 1, 0, 0], dtype=np.bool)
        res = utils.indices_above(a, 3)
        testing.assert_array_equal(desired, res)

    def test_extend_plus(self):
        a = np.array([1, 2, 3, 4, 3, 2, 1])
        desired = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.bool)
        res = utils.indices_above(a, 3, extend_window=1, extend_dir='+')
        testing.assert_array_equal(desired, res)

    def test_extend_minus(self):
        a = np.array([1, 2, 3, 4, 3, 2, 1])
        desired = np.array([0, 1, 1, 1, 1, 1, 0], dtype=np.bool)
        res = utils.indices_above(a, 3, extend_window=1, extend_dir='-')
        testing.assert_array_equal(desired, res)

    def test_extend_minus_edge(self):
        a = np.array([1, 4, 4, 2, 2, 2, 1])
        desired = np.array([1, 1, 1, 1, 1, 0, 0], dtype=np.bool)
        res = utils.indices_above(a, 3, extend_window=2, extend_dir='-')
        testing.assert_array_equal(desired, res)


class TestIndexBelow:
    def test_no_extend(self):
        a = np.array([1, 2, 3, 4, 3, 2, 1])
        desired = np.array([1, 1, 0, 0, 0, 1, 1], dtype=np.bool)
        res = utils.indices_below(a, 3)
        testing.assert_array_equal(desired, res)

    def test_extend_plus(self):
        a = np.array([1, 2, 3, 4, 3, 2, 1])
        desired = np.array([1, 1, 1, 0, 1, 1, 1], dtype=np.bool)
        res = utils.indices_below(a, 3, extend_window=1, extend_dir='+')
        testing.assert_array_equal(desired, res)

    def test_extend_minus(self):
        a = np.array([1, 2, 3, 4, 3, 2, 1])
        desired = np.array([1, 0, 0, 0, 0, 0, 1], dtype=np.bool)
        res = utils.indices_below(a, 3, extend_window=1, extend_dir='-')
        testing.assert_array_equal(desired, res)


class TestRemoveAround:
    def test_no_extend(self):
        a = np.array([1, 2, 3, 4, 3, 2, 1])
        desired = np.array([1, 1, 1, 0, 1, 1, 1], dtype=np.bool)
        res = utils.remove_around(a, 3, 0)
        testing.assert_array_equal(desired, res)
    
    def test_center(self):
        a = np.array([1, 2, 3, 4, 3, 2, 1])
        desired = np.array([1, 1, 0, 0, 0, 1, 1], dtype=np.bool)
        res = utils.remove_around(a, 3, 1)
        testing.assert_array_equal(desired, res)

    def test_left(self):
        a = np.array([1, 2, 3, 4, 3, 2, 1])
        desired = np.array([0, 0, 1, 1, 1, 1, 1], dtype=np.bool)
        res = utils.remove_around(a, 0, 1)
        testing.assert_array_equal(desired, res)

    def test_right(self):
        a = np.array([1, 2, 3, 4, 3, 2, 1])
        desired = np.array([1, 1, 1, 1, 1, 0, 0], dtype=np.bool)
        res = utils.remove_around(a, len(a)-1, 1)
        testing.assert_array_equal(desired, res)

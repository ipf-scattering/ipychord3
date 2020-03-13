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

import numpy as np
from numpy import testing

from ipychord3 import sf


class TestExtendImage:
    def test_odd_center(self):
        """if the center is the middle of the pattern we should get the pattern back"""
        ar = np.arange(9).reshape((3,3))
        img = {'map': ar}
        ext = sf.extend_image(img, (1,1))
        testing.assert_array_equal(ar, ext['map'])

    def test_off_by_one_v_right(self):
        ar = np.array([[11, 12, 13], 
                       [21, 22, 23],
                       [31, 32, 33]])

        desired = np.array([[11, 12, 13, 0, 0], 
                            [21, 22, 23, 0, 0],
                            [31, 32, 33, 0, 0]])
        
        ext = sf.extend_image({'map': ar}, (2,1))
        testing.assert_array_equal(desired, ext['map'])

    def test_off_by_one_v_left(self):
        ar = np.array([[11, 12, 13], 
                       [21, 22, 23],
                       [31, 32, 33]])

        desired = np.array([[0, 0, 11, 12, 13], 
                            [0, 0, 21, 22, 23],
                            [0, 0, 31, 32, 33]])
        
        ext = sf.extend_image({'map': ar}, (0,1))
        testing.assert_array_equal(desired, ext['map'])

    def test_off_by_one_h_top(self):
        ar = np.array([[11, 12, 13], 
                       [21, 22, 23],
                       [31, 32, 33]])

        desired = np.array([[ 0,  0,  0],
                            [ 0,  0,  0],
                            [11, 12, 13], 
                            [21, 22, 23],
                            [31, 32, 33]])
        
        ext = sf.extend_image({'map': ar}, (1,0))
        testing.assert_array_equal(desired, ext['map'])

    def test_off_by_one_h_bottom(self):
        ar = np.array([[11, 12, 13], 
                       [21, 22, 23],
                       [31, 32, 33]])

        desired = np.array([[11, 12, 13], 
                            [21, 22, 23],
                            [31, 32, 33],
                            [ 0,  0,  0],
                            [ 0,  0,  0]])
        
        ext = sf.extend_image({'map': ar}, (1,2))
        testing.assert_array_equal(desired, ext['map'])

    def test_off_by_one_bottom_right(self):
        ar = np.array([[11, 12, 13], 
                       [21, 22, 23],
                       [31, 32, 33]])

        desired = np.array([[11, 12, 13, 0, 0], 
                            [21, 22, 23, 0, 0],
                            [31, 32, 33, 0, 0],
                            [ 0,  0,  0, 0, 0],
                            [ 0,  0,  0, 0, 0]])
        
        ext = sf.extend_image({'map': ar}, (2,2))
        testing.assert_array_equal(desired, ext['map'])


class TestHarmony:
    def test_22(self):
        ar = np.array([[11, 12, 13], 
                       [21, 22, 23],
                       [31, 32, 33]])

        desired = np.array([[11, 12, 13, 12, 11], 
                            [21, 22, 23, 22, 21],
                            [31, 32, 33, 32, 31],
                            [21, 22, 23, 22, 21],
                            [11, 12, 13, 12, 11]])
        
        ext = sf.harmony({'map': ar}, (2,2))
        testing.assert_array_equal(desired, ext['map'])
        testing.assert_array_equal([2,2], ext['beam_position'])

    def test_00(self):
        ar = np.array([[11, 12, 13], 
                       [21, 22, 23],
                       [31, 32, 33]])

        desired = np.array([[33, 32, 31, 32, 33],
                            [23, 22, 21, 22, 23],
                            [13, 12, 11, 12, 13], 
                            [23, 22, 21, 22, 23],
                            [33, 32, 31, 32, 33]])
        
        ext = sf.harmony({'map': ar}, (0,0))
        testing.assert_array_equal(desired, ext['map'])
        testing.assert_array_equal([2,2], ext['beam_position'])
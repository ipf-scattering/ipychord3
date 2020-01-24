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

import pytest

import os

import numpy as np
from numpy import testing

from ipychord3 import io

class TestFIOreader:
    @staticmethod
    @pytest.fixture(scope="class")
    def getData():
        dataFile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'data', 'test.fio')
        return io.FIOreader(dataFile)

    def test_begin(self):
        reader = self.getData()
        actual = reader.begin
        desired = '29-Apr-2016 18:43:17'
        assert actual == desired

    def test_end(self):
        reader = self.getData()
        actual = reader.end
        desired = '18:43:34'
        assert actual == desired

    def test_parameter_length(self):
        reader = self.getData()
        actual = len(reader.parameters)
        desired = 4
        assert actual == desired

    def test_parameter(self):
            reader = self.getData()
            actual = reader.parameters
            desired =  [('ENERGY',  13182),
                        ('ZOOM',  140.8021),
                        ('PRIM_BSX', -11.87507),
                        ('VM3', -123.3106)]

            for key, val in desired:
                assert key in actual.keys()
                testing.assert_almost_equal(reader[key], val)

    def test_data_header(self):
        reader = self.getData()
        actual = reader.data_header
        desired = ['POS', 'CM', 'EXP_VFC01', 'EXP_VFC02', 'EXP_VFC06', 'IPETRA']
        assert actual == desired

    def test_data(self):
        reader = self.getData()
        actual = reader.data
        desired = np.array([[0, 0, 0, 0.5, 34.5, 4.95305446884937e+13],
                            [15, 0.5, 0.5, 0.5, 0, 0.00000000000000e+00]])
        testing.assert_almost_equal(actual, desired)

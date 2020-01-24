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

"""This module handles special io for the experiments.
"""
import logging
import os
import datetime
import warnings

from pathlib import PurePosixPath
from collections import defaultdict

import numpy as np
import h5py

from scipy.signal import medfilt2d
from scipy.interpolate import griddata

from . import io

# setup logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class IRBISTextReader:
    '''Reader for text files from IRBIS

    If the file contains corrupted data, it will be interpolated.


    :param fname: file name of IRBIS file
    :type fname: string

    .. autoinstanceattribute:: parameters
        :annotation:

        dictionary of the parameters
    '''
    def __init__(self, fname):
        self.fname = fname

        self.parameters = {}
        self.parameters['title'] = os.path.basename(fname)

        header_lines = []
        with open(fname, 'r', encoding='iso-8859-1', newline=None) as f:
            for i, l in enumerate(f):
                if l.startswith('[Data]'):
                    self.didx = i
                    break
                header_lines.append(l)

        sidx = header_lines.index('[Settings]\n')
        pidx = header_lines.index('[Parameter]\n')

        self._parse_settings(header_lines[sidx+1:pidx-1])
        self._parse_parameters(header_lines[pidx+1:-1])

    @property
    def map(self):
        '''array of the image'''
        logger.debug("reading data for: %s" % self.fname)
        return self._parse_data(self.didx+1)

    def __read_range(self, data):
        return [float(i) for i in data.split(';')]

    def _parse_settings(self, data):
        d = [line.strip().split('=') for line in data]
        conv = defaultdict(lambda: str)
        conv['Version'] = int
        conv['ImageWidth'] = int
        conv['ImageHeight'] = int
        conv['ShotRange'] = self.__read_range
        conv['CalibRange'] = self.__read_range
        for i in d:
            self.parameters[i[0]] = conv[i[0]](i[1])

    def _parse_parameters(self, data):
        d = [line.strip().split('=') for line in data]

        # convert new names to old names
        rename_keys = {'RecTime': 'Rec.time', 'RecDate': 'Rec.date'}
        d_new = []
        for i in d:
            if i[0] in rename_keys.keys():
                d_new.append([rename_keys[i[0]], i[1]])
            else:
                d_new.append(i)
        d = d_new

        conv = defaultdict(lambda: str)
        conv['FrameIndex'] = int
        conv['ms'] = float
        conv['Distance'] = float
        conv['Env.temp'] = float
        conv['Rec.time'] = lambda x: x.replace(',', '.')
        for i in d:
            self.parameters[i[0]] = conv[i[0]](i[1])

    def _parse_data(self, skip):
        ar = np.genfromtxt(self.fname, skip_header=skip)
        if np.any(np.isnan(ar)):
            # interpolate nan's https://stackoverflow.com/a/39596856
            logger.info("found nan's, interpolating, the file is %s, nan's at %s" %(self.fname, str(np.argwhere(np.isnan(ar)))))
            x = np.arange(0, ar.shape[1])
            y = np.arange(0, ar.shape[0])
            array = np.ma.masked_invalid(ar)
            xx, yy = np.meshgrid(x, y)
            #get only the valid values
            x1 = xx[~array.mask]
            y1 = yy[~array.mask]
            newarr = array[~array.mask]
            ar = griddata((x1, y1), newarr.ravel(),
                                    (xx, yy),
                                    method='linear')
        return ar


    def get_data_dict(self):
        """returns a `dict` with all the parameters in `parameters` plus the the
        image under key `map`
        """
        ddict = dict()
        ddict.update(self.parameters)

        ddict['map'] = self.map

        return ddict


class PILCReader:
    """Read data recorded by PiLC at DESY

    :param fname: neme of the file
    :param data_map: dictionary mapping the PiLC names (``ADC1``, ``ADC2``) to a
    :param convert_to_voltage: convert data to voltage?

    :returns: dictionary of data mapped by ``data_map``
    """
    def __init__(self, fname, data_map, convert_to_voltage=False):
        self.fname = fname
        self.data_map = data_map
        self.convert_to_voltage = convert_to_voltage

    @staticmethod
    def to_voltage(x):
        """Convert counts to voltage

        :param x: array of counts
        :return: array with voltage values
        """
        return 10/(2**16 - 1) * x

    def get_data(self):
        """read the data from file
        """
        with h5py.File(self.fname, 'r') as  f:
            ret = dict()
            for k in self.data_map.keys():
                data = f[os.path.join('/Scan/data/', k)].value[1:]
                if self.convert_to_voltage:
                    data = self.to_voltage(data)
                ret[self.data_map[k]] = data
        return ret


def cbf_pattern_selector(cbfs, skip_begin=1, skip_end=1, atol_zeros=10):
    """Select CBFs with a mean intensity greater than zero.

    Additionally it is possible to skip non-zero pattern at the beginning and end.

    :param cbfs: list of CBF file names
    :param skip_begin: number of non-zero pattern to skip at the beginning
    :param skip_end: number of non-zero pattern to skip at the end
    :param atol_zero: the average intensity of the patter to still be consiered zero
    :return: ``selector``, ``selected_cbfs``

       * ``selector``: an array of bool's with length ``len(cbfs)``,
         ``True`` for a file to choose and ``False`` otherwise.
       * ``selected_cbfs``: a list with the selected file names
    """
    means = np.asarray([medfilt2d(io.CBFreader(cbf).map, 5).mean() for cbf in cbfs])
   # means = np.asarray([io.CBFreader(cbf).map.mean() for cbf in cbfs])
    selector = np.logical_not(np.isclose(means, np.zeros_like(means),atol=atol_zeros))
    idxs = np.nonzero(selector)[0]
    if skip_begin == 0:
        skip_idxs = []
    else:
        skip_idxs = idxs[:skip_begin]
    if skip_end != 0:
        skip_idxs = np.concatenate((skip_idxs, idxs[-skip_end:]))

    for idx in skip_idxs:
        selector[idx] = False

    selected_cbfs = []
    for path, select in zip(cbfs, selector):
        if select:
            selected_cbfs.append(path)
    return selector, selected_cbfs


def pilc_data_for_pattern(pattern_selector, beamstop, npnts_per_pattern, skip_pattern_begin):
    """Create a function to collect the PiLC data for the pattern.

    The beamstop signal is used to determine the opening of the fast shutter, at this point we will also have a non-zero pattern.

    For each pattern the data is averaged over  ``npnts_per_pattern``.

    .. note:: There is no ``skip_pattern_end`` or the like, as PiLC keeps recording after Pilatus has finished the
     acquisition and the fast shutter has been closed. This means we have much more data from PiLC as the number of
     pattern would suggest. Thus at the end all data is cut to match the number of pattern.

    :param pattern_selector: the selector returned by :py:func:`ipychord3.special_io.cbf_pattern_selector`
    :param beamstop: the beamstop signal from PiLC
    :param npnts_per_pattern: the number of points recorded by PiLC per pattern
    :param skip_pattern_begin: the number of pattern to skip at the beginning (should be the same value as``skip_begin`` for :py:func:`ipychord3.special_io.cbf_pattern_selector`)
    :return: a function to select the PiLC data for the pattern.

    Usage::

      pilc_selector = pilc_data_for_pattern(pattern_selector, beamstop, npnts_per_pattern, skip_pattern_begin)
      displacement = pilc_selector(DISPLACEMENT_FROM_PILC)
    """
    n_pattern = np.sum(pattern_selector)
    bs_null = beamstop[:3].mean()
    skip_pilc_begin = npnts_per_pattern * skip_pattern_begin

    selector = np.logical_not(np.isclose(beamstop - bs_null, np.zeros_like(bs_null), atol=1))
    idxs = np.nonzero(selector)[0]

    last_idx = 0
    if skip_pilc_begin == 0:
        skip_idxs = []
    else:
        skip_idxs = idxs[:skip_pilc_begin]

    for idx in skip_idxs:
        selector[idx] = False
        last_idx = idx

    nid = last_idx + 1 + (n_pattern * npnts_per_pattern)

    selector[last_idx + 1:nid] = True
    selector[nid:] = False

    bs = beamstop[selector]
    cutoff = bs.shape[0] % npnts_per_pattern

    if cutoff:
        bs = bs[:-cutoff]

    bs = np.sum(bs.reshape(-1, npnts_per_pattern), axis=1)

    if bs.shape[0] < n_pattern:
        raise RuntimeError("Number of pattern (%d) and data points (%d) doesn't match" % (n_pattern, bs.shape[0]))
    elif bs.shape[0] > n_pattern:
        strip = n_pattern
    else:
        strip = None

    def selector_function(x):
        xs = x[selector]
        if cutoff:
            xs = xs[:-cutoff]
        xs = np.sum(xs.reshape(-1, npnts_per_pattern), axis=1)/npnts_per_pattern
        return xs[:strip]

    return selector_function


class ThermoSelector:
    """Select a thermography image closest or interpolated to a given timestamp.

    :param thermos: list of files names of thermography images
    :param shift: time shift for the times (in seconds)

       ``timestamp = timestamp + shift``

       such that the timestamp of the thermo image is shifted to the timestamp of the cbf

    :param interpolate: interpolate data to timestamp, otherwise closest image is returned

    :returns: On call the object returns a thermography image as numpy array.

    Usage::

      thermo = ThermoSelector(THERMO_FILES, 99.8)
      thermo_image = thermo(TIMESTAMP)
    """
    def __init__(self, thermos, shift, interpolate=True):
        self.thermos = thermos
        self.interpolate = interpolate

        timestamps_thermo = []
        for i in self.thermos:
            t = IRBISTextReader(i)
            timestamps_thermo.append(datetime.datetime.strptime(
                "%sT%s" %(t.parameters['Rec.date'],
                          t.parameters['Rec.time']),
                '%d.%m.%YT%H:%M:%S.%f').timestamp())
        self.timestamps_thermo = np.asarray(timestamps_thermo) + shift

    def __call__(self, timestamp):
        """

        :param timestamp: timestamp
        :return: thermography image as numpy array
        """
        idx = np.argmin(np.abs(self.timestamps_thermo - timestamp))
        if not self.interpolate:
            logger.debug("thermo image not interpolated")
            logger.debug("the thermo image is: '%s'" %self.thermos[idx])
            return IRBISTextReader(self.thermos[idx]).map

        dt = timestamp - self.timestamps_thermo[idx]

        if dt <= 0:
            idx_0 = idx-1
            idx_1 = idx
        else:
            idx_0 = idx
            idx_1 = idx + 1

        logger.debug("thermo image is interpolated")
        logger.debug("the thermo image 0 is: '%s'" %self.thermos[idx_0])
        logger.debug("the thermo image 1 is: '%s'" %self.thermos[idx_1])

        if idx_1 >= len(self.timestamps_thermo):
            warnings.warn("There are not enough thermography images, will fill with zeros.")
            return np.zeros_like(IRBISTextReader(self.thermos[idx_0]).map)

        x_0 = self.timestamps_thermo[idx_0]
        y_0 = IRBISTextReader(self.thermos[idx_0]).map
        x_1 = self.timestamps_thermo[idx_1]
        y_1 = IRBISTextReader(self.thermos[idx_1]).map

        return y_0 + (timestamp - x_0) * (y_1 - y_0) / (x_1 - x_0)


class DataCollection:  # (Sequence):
    """Reader for the collected data.

    :param fname: filename

    The accessible data is ``thermo``, ``scattering``, ``background``, ``mechanics``, ``time``.
    Each of these parts of the data is represented by :py:class:`ipychord3.special_io.DataSubCollection`.

    Indexing the object will return a :py:class:`ipychord3.special_io.DataCollectionItem` containing all the data for the given index.

    Usage::

      data = DataCollection(FILENAME)
      for data_point in data:
         imshow(data_point['thermo']['map'])
         title('displacement = %3g' %data_point['mechanics']['displacemnet'])

      ## or
      for p in data.scattering:
         sf.show(p)
    """
    def __init__(self, fname):
        self._fname = fname

        self._f = h5py.File(self._fname, 'r')
        self._thermo = DataSubCollection(self._f, '/thermo')
        self._thermo.add_getitem_callback(self._title_filename_callback)
        self._scattering = DataSubCollection(self._f, '/scattering')
        self._sample = DataSubCollection(self._f, '/sample')
        self._scattering.add_getitem_callback(self._title_filename_callback)
        self._background = DataSubCollection(self._f, '/scattering/background')
        self._mechanics = DataSubCollection(self._f, '/mechanics')
        self._time = DataSubCollection(self._f, '/time')

    def __len__(self):
        return len(self._thermo)

    def __getitem__(self, item):
        ret = DataCollectionItem()

        ret['thermo'] = self._thermo[item]
        ret['scattering'] = self._scattering[item]
        ret['background'] = self._background[:]
        ret['mechanics'] = self._mechanics[item]
        ret['time'] = self._time[item]
        ret['sample'] = self._sample[item]

        return ret

    @property
    def thermo(self):
        """Get only the thermo data."""
        return self._thermo

    @property
    def scattering(self):
        """Get only the scattering data."""
        return self._scattering

    @property
    def sample(self):
        """Get only the sample data."""
        return self._sample

    @property
    def background(self):
        """Get only the background data."""
        return self._background

    @property
    def mechanics(self):
        """Get only the mechanics data."""
        return self._mechanics

    @property
    def time(self):
        """Get only the time data."""
        return self._time

    @staticmethod
    def _title_filename_callback(item, parameters):
        ## this is needed by sf.show
        if parameters:
            parameters['title'] = parameters['title'] + '_%05d' % item
            parameters['filename'] = parameters['title']
        return parameters

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._f.close()

    def close(self):
        """close file"""
        self._f.close()


class DataSubCollection:  # (Sequence):
    """Class the represents on part of the data.

    Indexing this class returns a dict with all the parameters and data sets.

    :param file: file object of the data
    :param data_path: which part of the data to read.

    .. autoinstanceattribute:: parameters
        :annotation:

        Dictionary of the parameters (Attributes of the hdf5 group).

    .. autoinstanceattribute:: DATASET_NAME
        :annotation:

        All data sets are also accessible via attributes with their name.

        The underlying `h5py Dataset
        <http://docs.h5py.org/en/latest/high/dataset.html>`_ is returned.
    """
    def __init__(self, file, data_path):
        self._file = file
        self._data_path = PurePosixPath(data_path)
        self._entry = self._file.require_group(str(self._data_path))

        self._getitem_callbacks = []

        self._dsets = []
        for i in self._entry.values():
            if isinstance(i, h5py.Dataset):
                self._dsets.append(os.path.basename(i.name))
            else:
                continue

        self.parameters = dict()
        for k in self._entry.attrs.keys():
            self.parameters[k] = self._entry.attrs[k]

        if self._dsets:
            self._len = self._file[str(self._data_path.joinpath(self._dsets[0]))].shape[0]
        else:
            self._len = 0

        for dset in self._dsets:
            setattr(self, dset, self._file[str(self._data_path.joinpath(dset))])

    def __len__(self):
        return self._len

    def add_getitem_callback(self, callback):
        """add a callback to change the parameters dict for each item

        :param callback: callback, with signature ``callback(item, parameters)``

        The callback should return a new parameters dict for the current item.
        """
        self._getitem_callbacks.append(callback)

    def __getitem__(self, item):
        ret = dict()
        ret.update(self.parameters)

        for callback in self._getitem_callbacks:
            ret = callback(item, ret)

        for dset in self._dsets:
            logger.debug("reading %s" %(dset.__repr__()))
            dshape = len(self._entry[dset].shape)

            if dshape == 1 or dshape == 3: # time series of single values or images
                logger.debug("reading time series data")
                ret[dset] = self._entry[dset][item]
            elif dshape == 2: # image
                logger.debug("reading single image")
                ret[dset] = self._entry[dset].value
            else:
                raise ValueError('Dimension of data set is too large, expected 1,2,3 got %d' %dshape)
        return ret

    def __iter__(self):
        return (self[i] for i in range(len(self)))


class DataCollectionItem(dict):
    '''A class representing one item of the data collection.

    It can be used like a dictionary but all the keys are also accessible via attributes.

    For example::

      data = DataCollection(FILENAME)
      data[0].sample

    is equivalent to::

      data = DataCollection(FILENAME)
      data[0]['sample']

    '''
    def __init__(self, *args, **kwargs):
        super(DataCollectionItem, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for key, value in arg.items():
                    self[key] = value

        if kwargs:
            for key, value in kwargs.items():
                self[key] = value

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DataCollectionItem, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DataCollectionItem, self).__delitem__(key)
        del self.__dict__[key]

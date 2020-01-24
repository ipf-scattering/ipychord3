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

import logging

import os
import re
import datetime

import numpy as np
import h5py

import fabio

# setup logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ['FIOreader', 'CBFreader',
           'writeh5', 'readh5', 'LambdaReader']


class FIOreader:
    """Reader for fio-files

    :param fname: file name of fio file
    :type fname: string
    :param auto_reverse_data: automatically reverse the data to scan range
    :type auto_reverse_data: bool


    .. autoinstanceattribute:: comment
       :annotation:

       the comment string in the file

    .. autoinstanceattribute:: parameters
       :annotation:

       dictionary of the parameters

    .. autoinstanceattribute:: data_header
       :annotation:

       list of the (striped) header of the data section

       it is contains only the names, e.g. ['POS', 'CM', ..., 'IPETRA']

    .. autoinstanceattribute:: data
       :annotation:

       ndarray of the data

    .. autoinstanceattribute:: begin
       :annotation:

       string begin time

    .. autoinstanceattribute:: end
       :annotation:

       string of end time

    .. autoinstanceattribute:: scanstring
       :annotation:

       the scanstring

    .. autoinstanceattribute:: scanrange
       :annotation:

       the scan range as list: '[start, stop, step]'
    """
    def __init__(self, fname, auto_reverse_data=False):
        self.fname = fname
        self.auto_reverse_data = auto_reverse_data
        self.comment = ''
        self.parameters = dict()
        self.data_header = list()
        self.data = list()
        self.begin = ''
        self.end = ''
        self.scanstring = ''
        self.scanrange = [0.0, 0.0, 0.0]

        self._read_file()

    def __getitem__(self, item):
        """access parameter values
        """
        return self.parameters[item]

    def _read_file(self):
        with open(self.fname, 'r') as f:
            lines = [l.strip() for l in f.readlines()]

        # read the comment
        comment_start = lines.index('%c') + 1
        comment_end = comment_start + lines[comment_start:].index('!')
        self.comment = ' '.join(lines[comment_start:comment_end])
        self._parse_comment(lines[comment_start:comment_end])

        # read parameters
        parameters_start = lines.index('%p') + 1
        parameters_end = parameters_start + lines[parameters_start:].index('!')
        for p in lines[parameters_start:parameters_end]:
            name, value = p.split('=')
            self.parameters[name.strip()] = float(value)

        # read data
        data_header_start = lines.index('%d') + 1
        data_header = []
        data = []
        for l in lines[data_header_start:]:
            if l.startswith('Col'):
                data_header.append(l.split())
            else:
                data.append([float(i) for i in l.split()])

        data_header = np.asarray(data_header)
        data_prefix = os.path.commonprefix(data_header[:, 2].tolist())
        data_head = [i[len(data_prefix):] for i in data_header[:, 2]]

        self.data_header = data_head
        self.data = np.asarray(data)
        if self.auto_reverse_data:
            if np.isclose(self.data[0, 0], self.scanrange[0]):
                # print('ordered', self.data[0, 0], self.scanrange[0])
                logger.debug('data is ordered')
            elif np.isclose(self.data[-1, 0], self.scanrange[0]):
                # print('reversed', self.data[-1, 0], self.scanrange[0])
                logger.debug('data is reversed')
                self.data = np.flipud(self.data)
            else:
                raise ValueError("Cannot determine if data is reversed or not.")

    def _parse_comment(self, comment):
        start_string = 'started at '
        start_pos = comment[0].index(start_string) + len(start_string)
        start, end = comment[0][start_pos:].split(',')
        end = end.strip()[len('ended'):].strip()
        self.begin = start
        self.end = end

        m = re.search(r'\A(.*)-Scan', comment[0])
        self.scanstring = m.group(0)

        scan_range_re = r'Name:\s+\w+\s+from\s+([-+]?\d*\.?\d+)\s+to\s+([-+]?\d*\.?\d+)\s+by\s+([-+]?\d*\.?\d+)\s+sampling\s+([-+]?\d*\.?\d+)\s+s'
        m = re.search(scan_range_re, self.comment)
        self.scanrange = [float(i) for i in m.groups()[:3]]


class CBFreader:
    '''Reader for CBF files

    :param fname: file name of cbf file
    :type fname: string

    .. autoinstanceattribute:: map
       :annotation:

        array of the image

    .. autoinstanceattribute:: parameters
        :annotation:

        dictionary of the parameters
    '''
    def __init__(self, fname):
        self.fname = fname
        self.map = None
        self.parameters = dict()

        self.file = fabio.open(self.fname)

        # define the CIF header entries we want
        self._CIF_HEADER_MAP = {'X-Binary-Size-Fastest-Dimension': int,
                                'X-Binary-Size-Second-Dimension': int,
                                'X-Binary-Size': int,
                                'X-Binary-Element-Type': str,
                                'conversions': str}

        self.parameters['filename'] = os.path.splitext(os.path.basename(self.file.filename))[0]
        self._parse_pilatus_header()
        self._parse_cif_header()
        self.map = np.asarray(self.file.data, dtype=np.float64)

    # from taken from https://www.dectris.com/technical_pilatus.html?file=tl_files/root/support/technical_notes/pilatus/Pilatus_CBF_Header_Specification.pdf
    _PILATUS_HEADER_MAP = {
        'Detector_identifier': ('Detector', [slice(0, None)], str),
        'Pixel_size': ('Pixel_size', [0, 3], float),
        'Silicon': ('Silicon', [2], float),
        'Exposure_time': ('Exposure_time', [0], float),
        'Exposure_period': ('Exposure_period', [0], float),
        'Tau': ('Tau', [0], float),
        'Count_cutoff': ('Count_cutoff', [0], int),
        'Threshold_setting': ('Threshold_setting', [0], float),
        'Gain_setting': ('Gain_setting', [0, 1], str),
        'N_excluded_pixels': ('N_excluded_pixels', [0], int),
        'Excluded_pixels': ('Excluded_pixels', [0], str),
        'Flat_field': ('Flat_field', [0], str),
        'Trim_file': ('Trim_file', [0], str),
        'Image_path': ('Image_path', [0], str),

        'Wavelength': ('Wavelength', [0], float),
        'Energy_range': ('Energy_range', [0, 1], float),
        'Detector_distance': ('Detector_distance', [0], float),
        'Detector_Voffset': ('Detector_Voffset', [0], float),
        'Beam_xy': ('Beam_xy', [0, 1], float),
        'Beam_x': ('Beam_xy', [0], float),
        'Beam_y': ('Beam_xy', [1], float),
        'Flux': ('Flux', [0], str),
        'Filter_transmission': ('Filter_transmission', [0], float),
        'Start_angle': ('Start_angle', [0], float),
        'Angle_increment': ('Angle_increment', [0], float),
        'Detector_2theta': ('Detector_2theta', [0], float),
        'Polarization': ('Polarization', [0], float),
        'Alpha': ('Alpha', [0], float),
        'Kappa': ('Kappa', [0], float),
        'Phi': ('Phi', [0], float),
        'Phi_increment': ('Phi_increment', [0], float),
        'Chi': ('Chi', [0], float),
        'Chi_increment': ('Chi_increment', [0], float),
        'Oscillation_axis': ('Oscillation_axis', [slice(0, None)], str),
        'N_oscillations': ('N_oscillations', [0], int),
        'Start_position': ('Start_position', [0], float),
        'Position_increment': ('Position_increment', [0], float),
        'Shutter_time': ('Shutter_time', [0], float),
        }

    def _parse_cif_header(self):
        '''parse the cif header
        '''
        header = self.file.header
        for k, v in self._CIF_HEADER_MAP.items():
            self.parameters[k] = v(header[k])

    def _find_timestamp(self, pilatus_header):
        timere = re.compile(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3,6}')
        for i, item in enumerate(pilatus_header):
            if timere.match(item):
                self.parameters['time'] = pilatus_header.pop(i)
                break

        if 'time' not in self.parameters.keys():
            raise ValueError('could not find time stamp')
        return pilatus_header

    def _parse_pilatus_header(self):
        pilatus_header = [i.strip('# ') for i in self.file.header['_array_data.header_contents'].split('\r\n')]

        pilatus_header = self._find_timestamp(pilatus_header)

        # remove ():=
        _pilatus_header = []
        for line in pilatus_header:
            for i in '():=,':
                line = line.replace(i, '')
            _pilatus_header.append(line)
        pilatus_header = _pilatus_header
        del _pilatus_header

        pop_idx = None
        for name in self._PILATUS_HEADER_MAP:
            for i, item in enumerate(pilatus_header):
                if item.startswith(self._PILATUS_HEADER_MAP[name][0]):
                    pop_idx = i
                    d = item[len(self._PILATUS_HEADER_MAP[name][0]):].split()
                    if self._PILATUS_HEADER_MAP[name][2] is str:
                        _d = []
                        for idx in self._PILATUS_HEADER_MAP[name][1]:
                            if isinstance(d[idx], str):
                                _d.append(d[idx])
                            else:
                                _d += d[idx]
                            self.parameters[name] = ' '.join(_d)
                    elif len(self._PILATUS_HEADER_MAP[name][1]) == 1:
                        self.parameters[name] = self._PILATUS_HEADER_MAP[name][2](d[self._PILATUS_HEADER_MAP[name][1][0]])
                    else:
                        _d = []
                        for idx in self._PILATUS_HEADER_MAP[name][1]:
                            _d.append(self._PILATUS_HEADER_MAP[name][2](d[idx]))
                        self.parameters[name] = tuple(_d)
                    break
            if pop_idx is not None:
                pilatus_header.pop(pop_idx)
                pop_idx = None

        if pilatus_header:
            raise ValueError('There are unknown parameters in the Pilatus header, %s' %str(pilatus_header))

    def get_data_dict(self):
        """mimics the `sf_pilcbfread` function

        returns a `dict` with all the parameters in `parameters` plus the the
        image under key `map`, additionally `mode`, `boxlen`, `width`, `height`
        and `title` are added to the dictionary (these keys are just aliases for other keys)
        """
        ddict = dict()
        ddict.update(self.parameters)
        ddict['title'] = ddict['filename']

        # convert size to nm
        ddict['boxlen'] = [i*1e9 for i in ddict['Pixel_size']]

        ddict['model'] = ddict['Detector_identifier']
        ddict['expot'] = ddict['Exposure_time']
        ddict['map'] = self.map

        return ddict


def writeh5(data, fname, dsets=['map'], reset_filename=True):
    '''write the data structure to a hdf5 file `fname`

    :param data: data dictionary
    :type data: dict
    :param fname: file name of h5 file
    :type fname: string
    :param dsets: list of datasets in data, the entry in this list will be
                  written as `Datasets` in the hdf5 file
    :type dsets: list
    :param reset_filename: should the filename in the data should be updated to the current filename

    The structure of the file is described in the documentation.
    '''

    if os.path.dirname(fname) and not os.path.isdir(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))

    with h5py.File(fname, 'w') as f:
        if reset_filename:
            data['filename'] = os.path.splitext(os.path.basename(fname))[0]
            data['title'] = data['filename']

        f.attrs['file_time'] = datetime.datetime.now().isoformat()
        f.attrs['HDF5_Version'] = h5py.version.hdf5_version
        f.attrs['h5py_version'] = h5py.version.version

        entry = f.create_group('entry')

        for k in data.keys():
            if k in dsets:
                entry.create_dataset(k, data=data[k], compression="gzip", compression_opts=9, shuffle=True)
            else:
                entry.attrs[k] = data[k]


def readh5(fname):
    '''read the data structure from hdf5 file `fname`

    :param fname: file name of h5 file
    :type fname: string
    :returns: dict of the data

    It is assumed that the structure of the file is as described in the documentation.
    '''

    with h5py.File(fname, 'r') as f:
        entry = f.require_group('entry')
        data = dict()
        for k in entry.attrs.keys():
            data[k] = entry.attrs[k]

        for i in entry.values():
            if isinstance(i, h5py.Dataset):
                data[os.path.basename(i.name)] = i[:]
            else:
                continue

    return data


class LambdaReader:
    """Reader for NXS files created by the Lambda detector

    The pattern can be accessed by indexing the object.
    Each item is a dictionary containing the ``map`` and additional information.

    :param fname: file name of nxs file
    :type fname: string
    :param apply_mask: whether or not to apply the mask
    :type apply_mask: bool

    Usage::

      lambda_data = LambdaReader(FANAME)
      for i in range(len(lambda_data)):
          p = lambda_data[i]
          sf.show(p, log=1, block=True)

    the dictionary contains some timing information:

    ``time = shutter_time * ImageNumber`` in milli seconds
    """
    def __init__(self, fname, apply_mask=True):
        self.fname = fname

        self.file = h5py.File(self.fname, 'r')

        self._data = self.file['/entry/instrument/detector/data']

        self._apply_mask = apply_mask
        self._apply_mask_pixels = ['dead pixel',
                                   'under responding pixel',
                                   'over responding pixel',
                                   'noisy pixel',
                                   'problematic pixel cluster']

        if self._apply_mask:
            self.mask = NexusMask(self.file['/entry/instrument/detector/pixel_mask'][:])
            self._mask = self.mask.get_mask(self.apply_mask_pixels)
        else:
            self.mask = None
            self._mask = None

        self._dead_time = self.file['/entry/instrument/detector/trigger_dead_time'][0]
        self._delay_time = self.file['/entry/instrument/detector/trigger_delay_time'][0]
        self.shutter_time = self.file['/entry/instrument/detector/collection/shutter_time'].value[0]

        if isinstance(self.file.attrs['file_time'], (bytes, str) ):
            self._ftime = self.file.attrs['file_time'].decode("utf-8") 
            self._basename = os.path.splitext(os.path.basename(self.file.attrs['file_name']))[0].decode("utf-8") 
        else:
            self._ftime = self.file.attrs['file_time'][0]
            self._basename = os.path.splitext(os.path.basename(self.file.attrs['file_name'][0]))[0]


    @property
    def data(self):
        '''the pattern data'''
        return self._data

    @property
    def apply_mask(self):
        """whether or not to apply the mask to the patterns"""
        return self._apply_mask

    @apply_mask.setter
    def apply_mask(self, vaule):
        """apply the mask `[True|False]`"""
        self._apply_mask = vaule

    @property
    def apply_mask_pixels(self):
        """list of pixel names that should be masked

        default is: ``['dead pixel',
        'under responding pixel',
        'over responding pixel',
        'noisy pixel',
        'problematic pixel cluster']``
        """
        return self._apply_mask_pixels

    @apply_mask_pixels.setter
    def apply_mask_pixels(self, value):
        if self.mask is None:
            return
        self._apply_mask_pixels = value
        self._mask = self.mask.get_mask(self._apply_mask_pixels)

    def __getitem__(self, item):
        """get the pattern
        """
        logger.debug('getting item: %d' %item)
        img = self._data[item][:]
        if self.apply_mask:
            logger.debug('applying mask')
            img[self._mask] = 0

        p = dict()
        p['map'] = img
        p['title'] = self._basename + '_%05d' % item
        p['filename'] = p['title']
        p['boxlen'] = [i * 1000 for i in [self.file['/entry/instrument/detector/x_pixel_size'][0],
                                          self.file['/entry/instrument/detector/y_pixel_size'][0]]]
        p['trigger_dead_time'] = self._dead_time
        p['trigger_delay_time'] = self._delay_time
        p['shutter_time'] = self.shutter_time
        p['time'] = item*(self.shutter_time)
        p['time stamp'] = self._ftime + '_(this is the time of the first image in the sequence)'
        return p

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return (self[i] for i in range(len(self.data)))

    def close(self):
        """close file"""
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()


class NexusMask:
    """Handle the Pixel mask

    :param mask: the bitmask in ``/entry/instrument/detector/pixel_mask``
    """
    def __init__(self, mask):
        self.mask = mask

        # TODO: the whole map is ambiguous as there are many undefined pixels
        self.__old_mask_map = (('gap pixel', 2 ** 0),
                               ('dead pixel', 2 ** 1),
                               ('under responding pixel', 2 ** 2),
                               ('over responding pixel', 2 ** 3),
                               ('noisy pixel', 2 ** 4),
                               ('undefined', 2 ** 5),
                               ('problematic pixel cluster', 2 ** 6),
                               ('undefined', 2 ** 7),
                               ('user defined mask', 2 ** 8),
                               ('undefined', 2 ** 9),
                               ('undefined', 2 ** 10),
                               ('undefined', 2 ** 11),
                               ('undefined', 2 ** 12),
                               ('undefined', 2 ** 13),
                               ('undefined', 2 ** 14),
                               ('undefined', 2 ** 15),
                               ('undefined', 2 ** 16),
                               ('undefined', 2 ** 17),
                               ('undefined', 2 ** 18),
                               ('undefined', 2 ** 19),
                               ('undefined', 2 ** 20),
                               ('undefined', 2 ** 21),
                               ('undefined', 2 ** 22),
                               ('undefined', 2 ** 23),
                               ('undefined', 2 ** 24),
                               ('undefined', 2 ** 25),
                               ('undefined', 2 ** 26),
                               ('undefined', 2 ** 27),
                               ('undefined', 2 ** 28),
                               ('undefined', 2 ** 29),
                               ('undefined', 2 ** 30),
                               ('virtual pixel', 2 ** 31))

        self._mask_map = {key: value for (key, value) in self.__old_mask_map}

    @property
    def mask_map(self):
        """a dictionary containing all the names and bits"""
        return self._mask_map

    def check_pixels(self):
        """check what bits are contained in the mask

        :returns: tuple of (mask_name, bit)
        """
        ret = []
        for name, bit in self._mask_map.items():
            if np.any(self.mask & bit):
                ret.append((name, int(np.log2(bit))))

        return tuple(ret)

    def get_mask(self, mask_pixels):
        """generate a mask

        :param mask_pixels: list of names that should be included in the mask
        :return: the mask
        """
        m = 0
        for i in mask_pixels:
            m |= self._mask_map[i]
        a = self.mask & m
        ret = a > 0
        if len(ret.shape) == 3:
            return ret[0]
        else:
            return ret
         

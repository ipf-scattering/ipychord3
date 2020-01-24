import datetime
from pprint import pprint

import numpy as np
import h5py

f = h5py.File('test_file.h5', 'w')

f.attrs['file_time'] = datetime.datetime.now().isoformat()
f.attrs['HDF5_Version'] = h5py.version.hdf5_version
f.attrs['h5py_version'] = h5py.version.version


entry = f.create_group('entry')
entry.attrs['box'] = [0.2, 0.3]
entry.attrs['detector'] = "Pilatus"

data = np.ones((10,30,20))
d = entry.create_dataset('map', data=data)

f.close()

data = dict()
with h5py.File('test_file.h5', 'r') as f:
    g = f.require_group('entry')
    for k in g.attrs.keys():
        data[k] = g.attrs[k]
    pattern = g['map'][:]
pprint(data)
print(pattern)

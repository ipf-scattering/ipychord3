import sys
import os
from pprint import pprint

sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt

from ipychord3 import io

#fname = 'apollo_1320_00001_00001.cbf'
fname = 'nr_cb_test_00001_00001.cbf'


cbf = io.CBFreader(fname)
cbf._read_file()
ddict = cbf.get_data_dict()

pprint(ddict)
print()

ddict['data2'] = np.ones((10,30,20))
io.writeh5(ddict, 'test.h5', dsets=['map', 'data2'])

data = io.readh5('test.h5')

pprint(data)
plt.imshow(data['map'], origin="lower")
print(data['map'].shape)
plt.show()

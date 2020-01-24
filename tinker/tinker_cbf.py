import sys
import os
from pprint import pprint

sys.path.insert(0, os.path.abspath('..'))

import matplotlib.pyplot as plt

from ipychord3 import io

#fname = 'apollo_1320_00001_00001.cbf'
fname = 'nr_cb_test_00001_00001.cbf'

#data = io.sf_pilcbfread300k(fname)

#plt.imshow(data['map'])
#plt.show()


cbf = io.CBFreader(fname)
cbf._read_file()
pprint(cbf.parameters)

ddict = cbf.get_data_dict()
plt.imshow(ddict['map'])
pprint(ddict)

plt.show()

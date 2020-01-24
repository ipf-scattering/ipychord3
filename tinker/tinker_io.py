
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

from ipychord3 import io

fname = r'D:\code\fio\apollo_755_00001.fio'
#fname = r'D:\code\fio\apollo_755_2_move_00001r1.fio'

reader = io.FIOreader(fname)

#print(reader.comment)

#print(reader['ENERGY'])

print(reader.data_header)
print(reader.data)

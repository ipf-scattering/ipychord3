from setuptools import setup, Extension

import numpy
import ipychord3

setup(
    name='ipychord3',
    version=ipychord3.__version__,
    description='package to analyze saxs and waxs experiments',
    packages=['ipychord3'],
    platforms='any',
    license='MIT',
    install_requires=[
                      'numpy',
                      'scipy',
                      'matplotlib',
                      'scikit-image',
                      'h5py',
                      'fabio'
                     ]
)

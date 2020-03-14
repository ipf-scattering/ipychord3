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
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    project_urls={
        'Documentation': 'https://ipychord3.readthedocs.io/',
        'Source': 'https://github.com/ipf-scattering/ipychord3/',
        'Tracker': 'https://github.com/ipf-scattering/ipychord3/issues',
    },
    python_requires='~=3.5',
    install_requires=[
                      'numpy',
                      'scipy',
                      'matplotlib',
                      'scikit-image',
                      'h5py',
                      'fabio'
                     ]
)

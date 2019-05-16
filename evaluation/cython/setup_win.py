__author__ = 'Mario Amrehn'

import sys
from distutils.core import setup
from distutils.extension import Extension
import shutil
from pathlib import Path

from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np


extra_compile_args = ['/EHsc', '/openmp', '/O2'][0:1]
include_dirs = [np.get_include()]

extra_link_args = ['/lgomp']

if sys.version_info[:2] > (3, 4):
    extra_link_args.extend(
        [
            '/LIBPATH:C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\include',
            '/LIBPATH:C:\Program Files\Microsoft SDKs\Windows\\v7.1\Include',
            #'/LIBPATH:C:\Program Files (x86)\Windows Kits\\10\Include',
            '/LIBPATH:C:\Program Files (x86)\Windows Kits\\10\Lib\\10.0.10240.0\\ucrt\\x64',
            '/LIBPATH:C:\Program Files (x86)\Windows Kits\\10\Include\\10.0.10240.0\\ucrt'
        ]
    )
    include_dirs.extend(
        [
            'C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\include',
            'C:\Program Files\Microsoft SDKs\Windows\\v7.1\Include',
            #'C:\Program Files (x86)\Windows Kits\\10\Include',
            'C:\Program Files (x86)\Windows Kits\\10\Lib\\10.0.10240.0\\ucrt\\x64',
            'C:\Program Files (x86)\Windows Kits\\10\Include\\10.0.10240.0\\ucrt',
        ]
    )

extensions = [
    Extension('growcut_cy',  # The extension's name
              ['growcut_cy.pyx'],  # The Cython source and additional C/C++ source files
              language='c++',
              # sources=["Rectangle.cpp"],  # Additional source file(s)
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    ]

setup(
    include_dirs=include_dirs,
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extensions, annotate=True)
)

# Call:
# python setup_win.py build_ext --inplace --compiler=msvc

# PATH=$PATH:$HOME/miniconda3/bin

so_file = next(next(Path('./build').glob('lib.win*')).glob('growcut_cy.*'))
shutil.copy2(src=so_file, dst=so_file.parents[2].joinpath(so_file.name))

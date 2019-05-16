__author__ = 'Mario Amrehn'

from distutils.core import setup
from distutils.extension import Extension
import shutil
from pathlib import Path

from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np

include_dirs = [np.get_include()]

extensions = [
    Extension('growcut_cy',  # The extension's name
              ['growcut_cy.pyx'],  # The Cython source and additional C/C++ source files
              language='c++',
              extra_compile_args=['-fopenmp', '-O3'],
              extra_link_args=['-fopenmp', '-lgomp']),
    ]

setup(
    include_dirs=include_dirs,
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extensions, annotate=True)
)

# Call:
# python setup_gnu.py build_ext --inplace --compiler=msvc

# PATH=$PATH:$HOME/miniconda3/bin

so_file = next(next(Path('./build').glob('lib.linux*')).glob('growcut_cy.*.so'))
shutil.copy2(src=so_file, dst=so_file.parents[2].joinpath(so_file.name))

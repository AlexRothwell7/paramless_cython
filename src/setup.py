from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name="Paramless",
    ext_modules=cythonize(
        'paramless_cython.pyx'), include_dirs=[numpy.get_include()]
)

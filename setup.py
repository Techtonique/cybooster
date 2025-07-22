from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import sys
import os

# Set the source file
ext_modules = [
    Extension(
        name="_boosterc",
        sources=["cybooster/_boosterc.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

# Cross-platform setup handling
setup(
    name="cybooster",
    version="0.1.0",
    ext_modules=cythonize(ext_modules),
    packages=["cybooster"],
    install_requires=["cython", "numpy", "jax", "jaxlib"],
    # Ensure that the appropriate build tools are installed
    setup_requires=["cython", "numpy", "jax", "jaxlib"],  # Ensure setup tools install Cython and NumPy
)

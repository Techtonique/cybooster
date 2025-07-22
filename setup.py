import os
import sys
import platform
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
from pathlib import Path

# Define base path
# Force relative paths
here = Path(__file__).parent
os.chdir(here)

# Get version from package
try:
    from cybooster import __version__
except ImportError:
    __version__ = "0.1.0"  # fallback version

# Platform-specific configurations
extra_compile_args = []
extra_link_args = []
define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

if sys.platform == 'darwin':
    # macOS specific settings
    extra_compile_args.extend(['-stdlib=libc++', '-mmacosx-version-min=10.15'])
    extra_link_args.extend(['-stdlib=libc++', '-mmacosx-version-min=10.15'])
    
    # Handle ARM64 architecture
    if platform.machine() == 'arm64':
        extra_compile_args.extend(['-arch', 'arm64'])
        extra_link_args.extend(['-arch', 'arm64'])
    else:
        extra_compile_args.extend(['-arch', 'x86_64'])
        extra_link_args.extend(['-arch', 'x86_64'])

elif sys.platform == 'win32':
    if platform.machine().endswith('ARM64'):
        extra_compile_args.extend(['/ARM64'])
        extra_link_args.extend(['/ARM64'])
    else:
        extra_compile_args.extend(['/EHsc', '/O2'])
        extra_link_args.extend(['/OPT:REF', '/OPT:ICF'])

else:  # Linux

    try:
        import subprocess
        subprocess.check_call(['gcc', '-v'])
        extra_compile_args.extend(['-O3', '-fopenmp'])
        extra_link_args.extend(['-fopenmp'])
    except:
        extra_compile_args.extend(['-O3'])

# Set the source file using pathlib for cross-platform path handling
ext_modules = [
    Extension(
        name="cybooster._boosterc",  # Use full dotted path
        sources=[str(here / "cybooster" / "_boosterc.pyx")],
        include_dirs=[numpy.get_include()],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c",
    )
]

setup(
    name="cybooster",
    version=__version__,
    author="T. Moudiki",
    author_email="thierry.moudiki@gmail.com",
    description="A high-performance gradient boosting implementation using Cython",
    long_description=(here / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/Techtonique/cybooster",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    license="BSD-3-Clause",
    keywords="gradient boosting cython machine learning",
    packages=["cybooster"],
    package_dir={"": str(here)},  # Important for in-place builds
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            'language_level': "3",
            'embedsignature': True,
        },
    ),
    install_requires=[
        "cython>=0.29.0",
        "numpy>=1.20.0",
        "jax>=0.3.0",
        "jaxlib>=0.3.0",
        "scipy>=1.6.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.50.0",
        "pandas>=1.1.0",
    ],
    python_requires=">=3.9",
    package_data={
        'cybooster': ['*.pxd', '*.pyx'],  # Include all necessary files
    },
    zip_safe=False,  # Important for C extensions
)
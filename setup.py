import os
import platform
import shutil
import sys
from packaging import version
from pathlib import Path
from setuptools import Command, Extension, setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        # Mark as platform-specific wheel
        self.root_is_pure = False
        
    def get_tag(self):
        python, abi, plat = _bdist_wheel.get_tag(self)
        # Replace 'linux_x86_64' with 'manylinux' tag
        if plat.startswith('linux'):
            plat = 'manylinux2014_x86_64'
        return python, abi, plat

CYTHON_MIN_VERSION = version.parse("3.0.10")

__version__ = "0.1.4"

class clean(Command):
    user_options = [("all", "a", "")]
    
    def initialize_options(self):
        self.all = True
        self.delete_dirs = []
        self.delete_files = []
        
        for root, dirs, files in os.walk("cybooster"):
            root = Path(root)
            for d in dirs:
                if d == "__pycache__":
                    self.delete_dirs.append(root / d)
            
            if "__pycache__" in root.name:
                continue
                
            for f in (root / x for x in files):
                ext = f.suffix
                if ext == ".pyc" or ext == ".so":
                    self.delete_files.append(f)
                if ext in (".c", ".cpp"):
                    source_file = f.with_suffix(".pyx")
                    if source_file.exists():
                        self.delete_files.append(f)
        
        build_path = Path("build")
        if build_path.exists():
            self.delete_dirs.append(build_path)
    
    def finalize_options(self):
        pass
    
    def run(self):
        for delete_dir in self.delete_dirs:
            shutil.rmtree(delete_dir)
        for delete_file in self.delete_files:
            delete_file.unlink()

EXTENSIONS = {
    "_boosterc": {"sources": ["cybooster/_boosterc.pyx"]},
}

def get_module_from_sources(sources):
    for src_path in map(Path, sources):
        if src_path.suffix == ".pyx":
            return ".".join(src_path.parts[:-1] + (src_path.stem,))
    raise ValueError(f"Could not find module from sources: {sources!r}")

def _check_cython_version():
    message = f"Please install Cython with a version >= {CYTHON_MIN_VERSION}"
    try:
        import Cython
    except ModuleNotFoundError:
        raise ModuleNotFoundError(message)
    
    if version.parse(Cython.__version__) < CYTHON_MIN_VERSION:
        message += f" The current version is {Cython.__version__} in {Cython.__path__}."
        raise ValueError(message)

def cythonize_extensions(extensions):
    _check_cython_version()
    from Cython.Build import cythonize
    
    directives = {
        "language_level": "3",
        "embedsignature": True,
        "boundscheck": False,
        "wraparound": False
    }
    
    macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    
    for ext in extensions:
        if ext.define_macros is None:
            ext.define_macros = macros
        else:
            ext.define_macros += macros
    
    return cythonize(extensions, compiler_directives=directives)

def get_extensions():
    import numpy
    
    numpy_includes = [numpy.get_include()]
    extensions = []
    
    for config in EXTENSIONS.values():
        name = get_module_from_sources(config["sources"])
        include_dirs = numpy_includes + config.get("include_dirs", [])
        
        # Platform-specific compile args
        extra_compile_args = []
        if sys.platform == "darwin":
            extra_compile_args.extend(["-stdlib=libc++", "-mmacosx-version-min=10.15"])
            if platform.machine() == "arm64":
                extra_compile_args.extend(["-arch", "arm64"])
            else:
                extra_compile_args.extend(["-arch", "x86_64"])
        elif sys.platform == "win32":
            extra_compile_args.extend(["/EHsc", "/O2"])
        
        ext = Extension(
            name=name,
            sources=config["sources"],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            language="c",
        )
        extensions.append(ext)
    
    if "sdist" not in sys.argv and "clean" not in sys.argv:
        extensions = cythonize_extensions(extensions)
    
    return extensions

if __name__ == "__main__":
    setup(
        ext_modules=get_extensions(),
        version=__version__,
        zip_safe=False,
        cmdclass={"clean": clean, 'bdist_wheel': bdist_wheel},
        options={
            'bdist_wheel': {
                'universal': False,  # Platform-specific wheel
                'plat_name': 'manylinux2014_x86_64' if sys.platform == 'linux' else None,
            }
        },
        platforms=['Linux', 'MacOS', 'Windows'],
    )
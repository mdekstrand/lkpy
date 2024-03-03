from setuptools import Extension, setup

try:
    from Cython.Build import cythonize

    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

EXT_SPECS = {"lenskit.util.kvp": None}


def _make_extension(name: str, opts: None) -> Extension:
    path = name.replace(".", "/")
    if USE_CYTHON:
        path += ".pyx"
    else:
        path += ".c"
    return Extension(name, [path])


EXTENSIONS = [_make_extension(ext, opts) for (ext, opts) in EXT_SPECS.items()]
if USE_CYTHON:
    EXTENSIONS = cythonize(EXTENSIONS)
setup(ext_modules=EXTENSIONS)

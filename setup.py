import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

cmdclass = {}
ext_modules = [
    Extension(
        "cython_clip_resize_mask",
        ["clip_resize_mask.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function"],
    )
]
cmdclass.update({'build_ext': build_ext})

setup(
    name='python_sds',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    include_dirs=[np.get_include()]
)

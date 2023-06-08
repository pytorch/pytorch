from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os 

inductor_module = CppExtension(
    'inductor_module', 
    sources=['inductor_module.cpp'],
    extra_compile_args=['-std=c++17'],
    library_dirs=['aot_inductor_output'],
    libraries=[os.path.join(os.path.dirname(os.path.realpath(__file__)))]
)

setup(
    name='inductor_module',
    version='0.0.1',
    ext_modules=[inductor_module],
    cmdclass={
        'build_ext': BuildExtension
    }
)

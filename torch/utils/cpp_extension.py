import os
import re
import subprocess
import sys
import warnings
from setuptools.command.build_ext import build_ext

MINIMUM_GCC_VERSION = (4, 9)
ABI_INCOMPATIBILITY_WARNING = '''
Your compiler ({}) may be ABI-incompatible with PyTorch.
Please use a compiler that is ABI-compatible with GCC 4.9 and above.
See https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html.'''


def check_compiler_abi_compatibility(compiler):
    '''
    Verifies that the given compiler is ABI-compatible with PyTorch.

    Arguments:
        compiler (str): The compiler executable name to check (e.g. 'g++')

    Returns:
        False if the compiler is (likely) ABI-incompatible with PyTorch,
        else True.
    '''
    try:
        info = subprocess.check_output('{} --version'.format(compiler).split())
    except Exception:
        _, error, _ = sys.exc_info()
        warnings.warn('Error checking compiler version: {}'.format(error))
    else:
        info = info.decode().lower()
        if 'gcc' in info:
            # Sometimes the version is given as "major.x" instead of semver.
            version = re.search(r'(\d+)\.(\d+|x)', info)
            if version is not None:
                major, minor = version.groups()
                minor = 0 if minor == 'x' else int(minor)
                if (int(major), minor) >= MINIMUM_GCC_VERSION:
                    return True
                else:
                    # Append the detected version for the warning.
                    compiler = '{} {}'.format(compiler, version.group(0))

    warnings.warn(ABI_INCOMPATIBILITY_WARNING.format(compiler))
    return False


class BuildExtension(build_ext):
    """A custom build extension for adding compiler-specific options."""

    def build_extensions(self):
        # On some platforms, like Windows, compiler_cxx is not available.
        if hasattr(self.compiler, 'compiler_cxx'):
            compiler = self.compiler.compiler_cxx[0]
        else:
            compiler = os.environ.get('CXX', 'c++')
        check_compiler_abi_compatibility(compiler)
        for extension in self.extensions:
            extension.extra_compile_args = ['-std=c++11']
        build_ext.build_extensions(self)


def include_paths():
    here = os.path.abspath(__file__)
    torch_path = os.path.dirname(os.path.dirname(here))
    return [os.path.join(torch_path, 'lib', 'include')]

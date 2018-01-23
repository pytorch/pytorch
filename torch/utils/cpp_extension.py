import os.path

from setuptools.command.build_ext import build_ext


class BuildExtension(build_ext):
    """A custom build extension for adding compiler-specific options."""

    def build_extensions(self):
        for extension in self.extensions:
            extension.extra_compile_args = ['-std=c++11']
        build_ext.build_extensions(self)


def include_paths():
    here = os.path.abspath(__file__)
    torch_path = os.path.dirname(os.path.dirname(here))
    return [os.path.join(torch_path, 'lib', 'include')]

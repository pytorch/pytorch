from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from distutils.spawn import find_executable
from distutils import sysconfig, log
import setuptools
import setuptools.command.build_py
import setuptools.command.develop
import setuptools.command.build_ext

from collections import namedtuple
from contextlib import contextmanager
import glob
import os
import multiprocessing
import shlex
import subprocess
import sys
from textwrap import dedent

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
SRC_DIR = os.path.join(TOP_DIR, 'caffe2')
CMAKE_BUILD_DIR = os.path.join(TOP_DIR, '.setuptools-cmake-build')

install_requires = []
setup_requires = []
tests_require = []

################################################################################
# Pre Check
################################################################################

CMAKE = find_executable('cmake')
assert CMAKE, 'Could not find "cmake" executable!'
NINJA = find_executable('ninja')
MAKE = find_executable('make')
assert NINJA or MAKE, \
    'Could not find neither "ninja" nor "make" executable!'

################################################################################
# utils functions
################################################################################


@contextmanager
def cd(path):
    if not os.path.isabs(path):
        raise RuntimeError('Can only cd to absolute path, got: {}'.format(path))
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)

################################################################################
# Version
################################################################################

try:
    git_version = subprocess.check_output(['git', 'describe', '--tags', 'HEAD'],
                                          cwd=TOP_DIR).decode('ascii').strip()
except (OSError, subprocess.CalledProcessError):
    git_version = None

with open(os.path.join(SRC_DIR, 'VERSION_NUMBER')) as version_file:
    VersionInfo = namedtuple('VersionInfo', ['version', 'git_version'])(
        version=version_file.read().strip(),
        git_version=git_version
    )

################################################################################
# Customized commands
################################################################################


class Caffe2Command(setuptools.Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass


class create_version(Caffe2Command):
    def run(self):
        with open(os.path.join(SRC_DIR, 'version.py'), 'w') as f:
            f.write(dedent('''
            version = '{version}'
            git_version = '{git_version}'
            '''.format(**dict(VersionInfo._asdict()))))


class cmake_build(Caffe2Command):
    """
    Compiles everything when `python setup.py build` is run using cmake.

    Custom args can be passed to cmake by specifying the `CMAKE_ARGS`
    environment variable. E.g. to build without cuda support run:
        `CMAKE_ARGS=-DUSE_CUDA=Off python setup.py build`

    The number of CPUs used by `make`/`ninja` can be specified by passing
    `-j<ncpus>` to `setup.py build`.  By default all CPUs are used.
    """
    user_options = [
        (str('jobs='), str('j'),
            str('Specifies the number of jobs to use with make or ninja'))
    ]

    built = False

    def initialize_options(self):
        self.jobs = multiprocessing.cpu_count()

    def finalize_options(self):
        self.jobs = int(self.jobs)

    def run(self):
        if cmake_build.built:
            return
        cmake_build.built = True

        if not os.path.exists(CMAKE_BUILD_DIR):
            os.makedirs(CMAKE_BUILD_DIR)

        with cd(CMAKE_BUILD_DIR):
            # configure
            cmake_args = [
                find_executable('cmake'),
                '-DBUILD_SHARED_LIBS=OFF',
                '-DPYTHON_EXECUTABLE:FILEPATH={}'.format(sys.executable),
                '-DPYTHON_INCLUDE_DIR={}'.format(sysconfig.get_python_inc()),
                '-DBUILD_TEST=OFF',
                '-DBUILD_BENCHMARK=OFF',
                '-DBUILD_BINARY=OFF',
            ]
            if NINJA:
                cmake_args.extend(['-G', 'Ninja'])
            if 'CMAKE_ARGS' in os.environ:
                extra_cmake_args = shlex.split(os.environ['CMAKE_ARGS'])
                # prevent crossfire with downstream scripts
                del os.environ['CMAKE_ARGS']
                log.info('Extra cmake args: {}'.format(extra_cmake_args))
            cmake_args.append(TOP_DIR)
            subprocess.check_call(cmake_args)

            build_args = [NINJA or MAKE]
            # control the number of concurrent jobs
            if self.jobs is not None:
                build_args.extend(['-j', str(self.jobs)])
            subprocess.check_call(build_args)


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        self.run_command('create_version')
        self.run_command('cmake_build')
        for d in ['caffe', 'caffe2']:
            for src in glob.glob(
                    os.path.join(CMAKE_BUILD_DIR, d, 'proto', '*.py')):
                dst = os.path.join(
                    TOP_DIR, os.path.relpath(src, CMAKE_BUILD_DIR))
                self.copy_file(src, dst)
        setuptools.command.build_py.build_py.run(self)


class build_ext(setuptools.command.build_ext.build_ext):
    def get_outputs(self):
        return [os.path.join(self.build_lib, d)
                for d in ['caffe', 'caffe2']]

    def run(self):
        self.run_command('cmake_build')
        setuptools.command.build_ext.build_ext.run(self)

    def build_extensions(self):
        i = 0
        while i < len(self.extensions):
            ext = self.extensions[i]
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)

            src = os.path.join(CMAKE_BUILD_DIR, filename)
            if not os.path.exists(src):
                del self.extensions[i]
            else:
                dst = os.path.join(os.path.realpath(self.build_lib), filename)
                self.copy_file(src, dst)
                i += 1


class develop(setuptools.command.develop.develop):
    def run(self):
        self.run_command('build_py')
        setuptools.command.develop.develop.run(self)


cmdclass = {
    'create_version': create_version,
    'cmake_build': cmake_build,
    'build_py': build_py,
    'build_ext': build_ext,
    'develop': develop,
}

################################################################################
# Extensions
################################################################################

ext_modules = [
    setuptools.Extension(
        name=str('caffe2.python.caffe2_pybind11_state'),
        sources=[]),
    setuptools.Extension(
        name=str('caffe2.python.caffe2_pybind11_state_gpu'),
        sources=[]),
]

################################################################################
# Packages
################################################################################

packages = setuptools.find_packages()

install_requires.extend(['protobuf',
                         'numpy',
                         'future',
                         'hypothesis',
                         'requests',
                         'scipy',
                         'six',])

################################################################################
# Test
################################################################################

setup_requires.append('pytest-runner')
tests_require.extend(['pytest-cov', 'hypothesis'])

################################################################################
# Final
################################################################################

setuptools.setup(
    name='caffe2',
    version=VersionInfo.version,
    description='Caffe2',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=packages,
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    author='jiayq',
    author_email='jiayq@fb.com',
    url='https://caffe2.ai',
)

#!/usr/bin/env python
#
# Copyright (c) 2016-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
#


# System imports
from __future__ import print_function
from glob import glob
import platform
import os

from distutils.core import *
from distutils.command.build import build

rundir = os.getcwd()
os.chdir(os.path.abspath(os.path.dirname(__file__)))

if any(i in os.environ for i in ["CC", "CXX"]):
    if "CC" not in os.environ:
        os.environ['CC'] = os.environ['CXX']
    if "CXX" not in os.environ:
        os.environ['CXX'] = os.environ['CC']
    if platform.system() == 'Linux':
        os.environ['LDSHARED'] = os.environ['CXX'] + " -shared"
    print("Environment specifies CC=%s CXX=%s"%(os.environ['CC'], os.environ['CXX']))

intel_compiler = os.getenv('CC', '') in ['icl', 'icpc', 'icc']
try:
    tbb_root = os.environ['TBBROOT']
    print("Using TBBROOT=", tbb_root)
except:
    tbb_root = '..'
    if not intel_compiler:
        print("Warning: TBBROOT env var is not set and Intel's compiler is not used. It might lead\n"
              "    !!!: to compile/link problems. Source tbbvars.sh/.csh file to set environment")
use_compiler_tbb = intel_compiler and tbb_root == '..'
if use_compiler_tbb:
    print("Using Intel TBB from Intel's compiler")
if platform.system() == 'Windows':
    if intel_compiler:
        os.environ['DISTUTILS_USE_SDK'] = '1'  # Enable environment settings in distutils
        os.environ['MSSdk'] = '1'
        print("Using compiler settings from environment")
    tbb_flag = ['/Qtbb'] if use_compiler_tbb else []
    tbb_flag += ['/EHsc'] # for Python 2
    compile_flags = ['/Qstd=c++11'] if intel_compiler else []
else:
    tbb_flag = ['-tbb'] if use_compiler_tbb else []
    compile_flags = ['-std=c++11', '-Wno-unused-variable']

_tbb = Extension("tbb._api", ["tbb/api.i"],
        include_dirs=[os.path.join(tbb_root, 'include')] if not use_compiler_tbb else [],
        swig_opts   =['-c++', '-O', '-threads'] + (  # add '-builtin' later
              ['-I' + os.path.join(tbb_root, 'include')] if not use_compiler_tbb else []),
        extra_compile_args=compile_flags + tbb_flag,
        extra_link_args=tbb_flag,
        libraries   =(['tbb'] if not use_compiler_tbb else []) +
                     (['irml'] if platform.system() == "Linux" else []),   # TODO: why do we need this?
        library_dirs=[ rundir,                                              # for custom-builds
                       os.path.join(tbb_root, 'lib', 'intel64', 'gcc4.4'),  # for Linux
                       os.path.join(tbb_root, 'lib'),                       # for MacOS
                       os.path.join(tbb_root, 'lib', 'intel64', 'vc_mt'),   # for Windows
                     ] if not use_compiler_tbb else [],
        language    ='c++',
        )


class TBBBuild(build):
    sub_commands = [  # define build order
        ('build_ext', build.has_ext_modules),
        ('build_py', build.has_pure_modules),
    ]


setup(  name        ="TBB",
        description ="Python API for Intel TBB",
        long_description="Python API to Intel(R) Threading Building Blocks library (Intel(R) TBB) "
                         "extended with standard Pool implementation and monkey-patching",
        url         ="https://software.intel.com/en-us/intel-tbb",
        author      ="Intel Corporation",
        author_email="inteltbbdevelopers@intel.com",
        license     ="Dual license: Apache or Proprietary",
        version     ="0.1",
        classifiers =[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Environment :: Plugins',
            'Intended Audience :: Developers',
            'Intended Audience :: System Administrators',
            'Intended Audience :: Other Audience',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 3',
            'Programming Language :: C++',
            'Topic :: System :: Hardware :: Symmetric Multi-processing',
            'Topic :: Software Development :: Libraries',
          ],
        keywords='TBB multiprocessing multithreading composable parallelism',
        ext_modules=[_tbb],
        packages=['tbb'],
        py_modules=['TBB'],
        cmdclass={'build': TBBBuild}
)

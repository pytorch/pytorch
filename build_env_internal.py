"""build_env_internal defines a bunch of helper functions for build_env.
"""

import multiprocessing
import os
import subprocess
import sys

def GetSubprocessOutput(commands):
  try:
    proc = subprocess.Popen(commands, stdout=subprocess.PIPE)
    out, err = proc.communicate()
  except OSError as err:
    print 'Cannot run command', commands, '. Return empty output.'
    return ''
  return out.strip()

def GetCompilerType(CC):
  # determine compiler type.
  _COMPILER_VERSION_STR = GetSubprocessOutput([CC, '--version'])
  if 'clang' in _COMPILER_VERSION_STR:
    return 'clang'
  elif ('g++' in _COMPILER_VERSION_STR or
        'Free Software Foundation' in _COMPILER_VERSION_STR):
    return 'g++'
  else:
    raise RuntimeError('Cannot determine C++ compiler type.')

def GetPythonIncludes():
  includes = GetSubprocessOutput(['python-config', '--includes'])
  # determine the numpy include directory. If any error happens, return
  # empty.
  try:
    import numpy.distutils
    includes += ' -I' + ' -I'.join(
      numpy.distutils.misc_util.get_numpy_include_dirs())
  except Exception as e:
    pass
  return includes

def GetPythonLibDirs():
  python_root = GetSubprocessOutput(['python-config', '--prefix'])
  lib_dir = os.path.join(python_root, 'lib')
  return ' -L' + lib_dir

""" build_env defines the general environment that we use to build.
"""

import multiprocessing
import os
import subprocess
import sys

def _GetSubprocessOutput(commands):
  try:
    proc = subprocess.Popen(commands, stdout=subprocess.PIPE)
    out, err = proc.communicate()
  except OSError as err:
    print 'Cannot run command', commands, '. Return empty output.'
    return ''
  return out.strip()

def _GetCompilerType(CC):
  # determine compiler type.
  _COMPILER_VERSION_STR = _GetSubprocessOutput([CC, '--version'])
  if 'clang' in _COMPILER_VERSION_STR:
    return 'clang'
  elif ('g++' in _COMPILER_VERSION_STR or
        'Free Software Foundation' in _COMPILER_VERSION_STR):
    return 'g++'
  else:
    raise RuntimeError('Cannot determine C++ compiler type.')


class Env(object):
  """Env is the class that stores all the build variables."""
  # Define the compile binary commands.
  CC = 'c++'
  MPICC = 'mpic++'
  LINK_BINARY = CC + ' -o'
  LINK_SHARED = CC + ' -shared -o'
  LINK_STATIC = 'ar rcs'
  # Protobuf constants
  PROTOC_BINARY = "protoc"

  if sys.platform == 'darwin':
    # For some reason, python on mac still recognizes the .so extensions...
    # So we will use .so here still.
    SHARED_LIB_EXT = '.so'
  elif sys.platform.startswith('linux'):
    SHARED_LIB_EXT = '.so'
  else:
    raise RuntimeError('Unknown system platform.')

  COMPILER_TYPE = _GetCompilerType(CC)

  #determine mpi include and mpi link flags.
  MPI_INCLUDES = _GetSubprocessOutput([MPICC, '--showme:incdirs']).split(' ')
  MPI_LIBDIRS = _GetSubprocessOutput([MPICC, '--showme:libdirs']).split(' ')
  MPI_LIBS = _GetSubprocessOutput([MPICC, '--showme:libs']).split(' ')
  if len(MPI_INCLUDES) == 1 and MPI_INCLUDES[0] == '':
    print ('MPI not found, so some libraries and binaries that use MPI will '
           'not compile correctly. If you would like to use those, you can '
           'install MPI on your machine. The easiest way to install on ubuntu '
           'is via apt-get, and on mac via homebrew.')
    # Set all values above to empty lists, so at least others will compile.
    MPI_INCLUDES = []
    MPI_LIBDIRS = []
    MPI_LIBS = []

  # Determine the CUDA directory.
  if os.path.exists('/usr/local/cuda'):
    CUDA_DIR = '/usr/local/cuda'
  else:
    raise RuntimeError('Cannot find Cuda directory.')
  NVCC = os.path.join(CUDA_DIR, 'bin', 'nvcc')
  NVCC_INCLUDES = [os.path.join(CUDA_DIR, 'include')]

  # Determine the NVCC link flags.
  if COMPILER_TYPE == 'clang':
    NVCC_LINKS = ('-rpath %s -L%s'
        % (os.path.join(CUDA_DIR, 'lib'), os.path.join(CUDA_DIR, 'lib')))
  elif COMPILER_TYPE == 'g++':
    NVCC_LINKS = ('-Wl,-rpath=%s -L%s'
        % (os.path.join(CUDA_DIR, 'lib64'), os.path.join(CUDA_DIR, 'lib64')))
  else:
    raise RuntimeError('Unknown compiler type to set nvcc link flags.')
  NVCC_LINKS += ' -l' + ' -l'.join([
      'cublas_static', 'curand_static', 'cuda', 'cudart_static', 'culibos'])
  if sys.platform.startswith('linux'):
    NVCC_LINKS += ' -l' + ' -l'.join(['rt', 'dl'])

  # NVCC C flags.
  NVCC_CFLAGS = ' '.join([
      # add cflags here.
      '-Xcompiler -fPIC',
      '-O2',
      '-std=c++11',
      '-gencode=arch=compute_30,code=sm_30',
  ])

  # Determine how the compiler deals with whole archives.
  if COMPILER_TYPE == 'clang':
    WHOLE_ARCHIVE_TEMPLATE = '-Wl,-force_load,%s'
  elif COMPILER_TYPE == 'g++':
    WHOLE_ARCHIVE_TEMPLATE = '-Wl,--whole-archive %s -Wl,--no-whole-archive'
  else:
    raise RuntimeError('Unknown compiler type to set whole-archive template.')

  # General cflags that should be added in all cc arguments.
  CFLAGS = ' '.join([
      # add cflags here.
      '-fPIC',
      '-DPIC',
      #'-O0',
      '-O2',
      #'-pg',
      '-DNDEBUG',
      '-msse',
      '-mavx',
      '-ffast-math',
      '-std=c++11',
      '-W',
      '-Wall',
      '-Wno-unused-parameter',
      '-Wno-sign-compare',
      #'-Wno-c++11-extensions',
  ])

  GENDIR = 'gen'
  # General include folders.
  INCLUDES = NVCC_INCLUDES + MPI_INCLUDES + [
      GENDIR,
      os.path.join(GENDIR, 'third_party'),
      os.path.join(GENDIR, 'third_party/include'),
      '/usr/local/include',
  ]
  INCLUDES = ' '.join(['-I' + s for s in INCLUDES])
  # Python
  INCLUDES += ' ' + _GetSubprocessOutput(['python-config', '--includes'])
  # General lib folders.
  LIBDIRS = MPI_LIBDIRS + [
      '/usr/local/lib',
  ]
  LIBDIRS = ' '.join(['-L' + s for s in LIBDIRS])
  # General link flags for binary targets
  LIBS = []
  LIBS = ' '.join(['-l' + s for s in LIBS])
  LINKFLAGS = ' '.join([
      # Add link flags here
      '-pthread',
      #'-pg',
  ]) + ' ' + LIBDIRS + ' ' + LIBS
  PYTHON_LIBS = [_GetSubprocessOutput(['python-config', '--ldflags'])]

  CPUS = multiprocessing.cpu_count()

  def __init__(self):
    """ENV is a singleton and should not be instantiated."""
    raise NotImplementedError(
        'Build system error: ENV should not be instantiated.')

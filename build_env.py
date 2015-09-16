""" build_env defines the general environment that we use to build.
"""

from build_env_internal import *

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
  # CUDA directory.
  CUDA_DIR = '/usr/local/cuda'
  # NVCC C flags.
  NVCC_CFLAGS = ' '.join([
      # add cflags here.
      '-Xcompiler -fPIC',
      '-O2',
      '-std=c++11',
      '-gencode arch=compute_20,code=sm_20',
      '-gencode arch=compute_20,code=sm_21',
      '-gencode arch=compute_30,code=sm_30',
      '-gencode arch=compute_35,code=sm_35',
      '-gencode arch=compute_50,code=sm_50',
      '-gencode arch=compute_50,code=compute_50',
  ])
  # General cflags that should be added in all cc arguments.
  CFLAGS = [
      # add cflags here.
      '-fPIC',
      '-DPIC',
      #'-O0',
      '-O2',
      '-g',
      #'-pg',
      '-DNDEBUG',
      #'-msse',
      #'-mavx',
      '-ffast-math',
      '-std=c++11',
      '-W',
      '-Wall',
      '-Wno-unused-parameter',
      '-Wno-sign-compare',
      #'-Wno-c++11-extensions',
  ]
  # The directory that contains the generated stuff.
  GENDIR = 'gen'
  # The include directories
  INCLUDES = [
      '/usr/local/include',
      '/usr/include',
  ]
  # The library directories.
  LIBDIRS = [
      '/usr/local/lib',
      '/usr/lib',
  ]
  LINKFLAGS = [
      # Add link flags here
      '-pthread',
      #'-pg',
  ]

  # Everything below should be automatically figured out. You most likely do not
  # need to change them.

  if sys.platform == 'darwin':
    # For some reason, python on mac still recognizes the .so extensions...
    # So we will use .so here still.
    SHARED_LIB_EXT = '.so'
  elif sys.platform.startswith('linux'):
    SHARED_LIB_EXT = '.so'
  else:
    raise RuntimeError('Unknown system platform.')

  COMPILER_TYPE = GetCompilerType(CC)

  #determine mpi include and mpi link flags.
  MPI_INCLUDES = GetSubprocessOutput([MPICC, '--showme:incdirs']).split(' ')
  MPI_LIBDIRS = GetSubprocessOutput([MPICC, '--showme:libdirs']).split(' ')
  MPI_LIBS = GetSubprocessOutput([MPICC, '--showme:libs']).split(' ')
  if len(MPI_INCLUDES) == 1 and MPI_INCLUDES[0] == '':
    print ('MPI not found, so some libraries and binaries that use MPI will '
           'not compile correctly. If you would like to use those, you can '
           'install MPI on your machine. The easiest way to install on ubuntu '
           'is via apt-get, and on mac via homebrew.')
    # Set all values above to empty lists, so at least others will compile.
    MPI_INCLUDES = []
    MPI_LIBDIRS = []
    MPI_LIBS = []
  # Try to figure out if mpi has cuda support
  OMPI_INFO = GetSubprocessOutput(['ompi_info', '--parsable', '--all'])
  OMPI_CUDA_SUPPORT = [
      r for r in OMPI_INFO.split('\n')
      if 'mpi_built_with_cuda_support:value' in r]
  if len(OMPI_CUDA_SUPPORT) == 1 and OMPI_CUDA_SUPPORT[0][-5:] == 'false':
    # If this is the case, we do not have cuda compiled in MPI, even if the
    # version forces so.
    CFLAGS.append('-DCAFFE2_FORCE_FALLBACK_CUDA_MPI')

  # Determine the CUDA directory.
  # TODO(Yangqing): find a way to automatically figure out where nvcc is.

  if not os.path.exists(CUDA_DIR):
    # Currently, we just print a warning.
    print ('Cannot find Cuda directory. NVCC will not run.')
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

  # Determine how the compiler deals with whole archives.
  if COMPILER_TYPE == 'clang':
    WHOLE_ARCHIVE_TEMPLATE = '-Wl,-force_load,%s'
  elif COMPILER_TYPE == 'g++':
    WHOLE_ARCHIVE_TEMPLATE = '-Wl,--whole-archive %s -Wl,--no-whole-archive'
  else:
    raise RuntimeError('Unknown compiler type to set whole-archive template.')

  CFLAGS = ' '.join(CFLAGS)
  INCLUDES += NVCC_INCLUDES + MPI_INCLUDES + [
      GENDIR,
      os.path.join(GENDIR, 'third_party'),
      os.path.join(GENDIR, 'third_party/include'),
  ]
  INCLUDES = ' '.join(['-I' + s for s in INCLUDES])
  # Python
  INCLUDES += ' ' + GetPythonIncludes()

  LIBDIRS += MPI_LIBDIRS
  LIBDIRS = ' '.join(['-L' + s for s in LIBDIRS])
  # Python
  LIBDIRS += ' ' + GetPythonLibDirs()
  # General link flags for binary targets
  LIBS = []
  LIBS = ' '.join(['-l' + s for s in LIBS])
  LINKFLAGS = ' '.join(LINKFLAGS) + ' ' + LIBDIRS + ' ' + LIBS
  PYTHON_LIBS = [GetSubprocessOutput(['python-config', '--ldflags'])]

  CPUS = multiprocessing.cpu_count()

  def __init__(self):
    """ENV is a singleton and should not be instantiated."""
    raise NotImplementedError(
        'Build system error: ENV should not be instantiated.')

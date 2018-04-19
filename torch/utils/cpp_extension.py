import copy
import glob
import imp
import os
import re
import setuptools
import subprocess
import sys
import sysconfig
import tempfile
import warnings

import torch
from .file_baton import FileBaton

from setuptools.command.build_ext import build_ext


def _find_cuda_home():
    '''Finds the CUDA install path.'''
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        if sys.platform == 'win32':
            cuda_home = glob.glob(
                'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
        else:
            cuda_home = '/usr/local/cuda'
        if not os.path.exists(cuda_home):
            # Guess #3
            try:
                which = 'where' if sys.platform == 'win32' else 'which'
                nvcc = subprocess.check_output(
                    [which, 'nvcc']).decode().rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
            except Exception:
                cuda_home = None
    return cuda_home


MINIMUM_GCC_VERSION = (4, 9)
MINIMUM_MSVC_VERSION = (19, 0, 24215)
ABI_INCOMPATIBILITY_WARNING = '''

                               !! WARNING !!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Your compiler ({}) may be ABI-incompatible with PyTorch!
Please use a compiler that is ABI-compatible with GCC 4.9 and above.
See https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html.

See https://gist.github.com/goldsborough/d466f43e8ffc948ff92de7486c5216d6
for instructions on how to install GCC 4.9 or higher.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                              !! WARNING !!
'''
CUDA_HOME = _find_cuda_home() if torch.cuda.is_available() else None


def check_compiler_abi_compatibility(compiler):
    '''
    Verifies that the given compiler is ABI-compatible with PyTorch.

    Arguments:
        compiler (str): The compiler executable name to check (e.g. ``g++``).
            Must be executable in a shell process.

    Returns:
        False if the compiler is (likely) ABI-incompatible with PyTorch,
        else True.
    '''
    try:
        check_cmd = '{}' if sys.platform == 'win32' else '{} --version'
        info = subprocess.check_output(
            check_cmd.format(compiler).split(), stderr=subprocess.STDOUT)
    except Exception:
        _, error, _ = sys.exc_info()
        warnings.warn('Error checking compiler version: {}'.format(error))
    else:
        info = info.decode().lower()
        if 'gcc' in info or 'g++' in info:
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
        elif 'Microsoft' in info:
            info = info.decode().lower()
            version = re.search(r'(\d+)\.(\d+)\.(\d+)', info)
            if version is not None:
                major, minor, revision = version.groups()
                if (int(major), int(minor),
                        int(revision)) >= MINIMUM_MSVC_VERSION:
                    return True
                else:
                    # Append the detected version for the warning.
                    compiler = '{} {}'.format(compiler, version.group(0))

    warnings.warn(ABI_INCOMPATIBILITY_WARNING.format(compiler))
    return False


class BuildExtension(build_ext):
    '''
    A custom :mod:`setuptools` build extension .

    This :class:`setuptools.build_ext` subclass takes care of passing the
    minimum required compiler flags (e.g. ``-std=c++11``) as well as mixed
    C++/CUDA compilation (and support for CUDA files in general).

    When using :class:`BuildExtension`, it is allowed to supply a dictionary
    for ``extra_compile_args`` (rather than the usual list) that maps from
    languages (``cxx`` or ``cuda``) to a list of additional compiler flags to
    supply to the compiler. This makes it possible to supply different flags to
    the C++ and CUDA compiler during mixed compilation.
    '''

    def build_extensions(self):
        self._check_abi()
        for extension in self.extensions:
            self._define_torch_extension_name(extension)

        # Register .cu and .cuh as valid source extensions.
        self.compiler.src_extensions += ['.cu', '.cuh']
        # Save the original _compile method for later.
        if self.compiler.compiler_type == 'msvc':
            self.compiler._cpp_extensions += ['.cu', '.cuh']
            original_compile = self.compiler.compile
            original_spawn = self.compiler.spawn
        else:
            original_compile = self.compiler._compile

        def unix_wrap_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            # Copy before we make any modifications.
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if _is_cuda_file(src):
                    nvcc = _join_cuda_home('bin', 'nvcc')
                    self.compiler.set_executable('compiler_so', nvcc)
                    if isinstance(cflags, dict):
                        cflags = cflags['nvcc']
                    cflags += ['--compiler-options', "'-fPIC'"]
                elif isinstance(cflags, dict):
                    cflags = cflags['cxx']
                # NVCC does not allow multiple -std to be passed, so we avoid
                # overriding the option if the user explicitly passed it.
                if not any(flag.startswith('-std=') for flag in cflags):
                    cflags.append('-std=c++11')

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # Put the original compiler back in place.
                self.compiler.set_executable('compiler_so', original_compiler)

        def win_wrap_compile(sources,
                             output_dir=None,
                             macros=None,
                             include_dirs=None,
                             debug=0,
                             extra_preargs=None,
                             extra_postargs=None,
                             depends=None):

            self.cflags = copy.deepcopy(extra_postargs)
            extra_postargs = None

            def spawn(cmd):
                orig_cmd = cmd
                # Using regex to match src, obj and include files

                src_regex = re.compile('/T(p|c)(.*)')
                src_list = [
                    m.group(2) for m in (src_regex.match(elem) for elem in cmd)
                    if m
                ]

                obj_regex = re.compile('/Fo(.*)')
                obj_list = [
                    m.group(1) for m in (obj_regex.match(elem) for elem in cmd)
                    if m
                ]

                include_regex = re.compile(r'((\-|\/)I.*)')
                include_list = [
                    m.group(1)
                    for m in (include_regex.match(elem) for elem in cmd) if m
                ]

                if len(src_list) >= 1 and len(obj_list) >= 1:
                    src = src_list[0]
                    obj = obj_list[0]
                    if _is_cuda_file(src):
                        nvcc = _join_cuda_home('bin', 'nvcc')
                        if isinstance(self.cflags, dict):
                            cflags = self.cflags['nvcc']
                        elif isinstance(self.cflags, list):
                            cflags = self.cflags
                        else:
                            cflags = []
                        cmd = [
                            nvcc, '-c', src, '-o', obj, '-Xcompiler',
                            '/wd4819', '-Xcompiler', '/MD'
                        ] + include_list + cflags
                    elif isinstance(self.cflags, dict):
                        cflags = self.cflags['cxx']
                        cmd += cflags
                    elif isinstance(self.cflags, list):
                        cflags = self.cflags
                        cmd += cflags

                return original_spawn(cmd)

            try:
                self.compiler.spawn = spawn
                return original_compile(sources, output_dir, macros,
                                        include_dirs, debug, extra_preargs,
                                        extra_postargs, depends)
            finally:
                self.compiler.spawn = original_spawn

        # Monkey-patch the _compile method.
        if self.compiler.compiler_type == 'msvc':
            self.compiler.compile = win_wrap_compile
        else:
            self.compiler._compile = unix_wrap_compile

        build_ext.build_extensions(self)

    def _check_abi(self):
        # On some platforms, like Windows, compiler_cxx is not available.
        if hasattr(self.compiler, 'compiler_cxx'):
            compiler = self.compiler.compiler_cxx[0]
        elif sys.platform == 'win32':
            compiler = os.environ.get('CXX', 'cl')
        else:
            compiler = os.environ.get('CXX', 'c++')
        check_compiler_abi_compatibility(compiler)

    def _define_torch_extension_name(self, extension):
        define = '-DTORCH_EXTENSION_NAME={}'.format(extension.name)
        if isinstance(extension.extra_compile_args, dict):
            for args in extension.extra_compile_args.values():
                args.append(define)
        else:
            extension.extra_compile_args.append(define)


def CppExtension(name, sources, *args, **kwargs):
    '''
    Creates a :class:`setuptools.Extension` for C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a C++ extension.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.

    Example:
        >>> from setuptools import setup
        >>> from torch.utils.cpp_extension import BuildExtension, CppExtension
        >>> setup(
                name='extension',
                ext_modules=[
                    CppExtension(
                        name='extension',
                        sources=['extension.cpp'],
                        extra_compile_args=['-g'])),
                ],
                cmdclass={
                    'build_ext': BuildExtension
                })
    '''
    include_dirs = kwargs.get('include_dirs', [])
    include_dirs += include_paths()
    kwargs['include_dirs'] = include_dirs

    if sys.platform == 'win32':
        library_dirs = kwargs.get('library_dirs', [])
        library_dirs += library_paths()
        kwargs['library_dirs'] = library_dirs

        libraries = kwargs.get('libraries', [])
        libraries.append('ATen')
        libraries.append('_C')
        kwargs['libraries'] = libraries

    kwargs['language'] = 'c++'
    return setuptools.Extension(name, sources, *args, **kwargs)


def CUDAExtension(name, sources, *args, **kwargs):
    '''
    Creates a :class:`setuptools.Extension` for CUDA/C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a CUDA/C++
    extension. This includes the CUDA include path, library path and runtime
    library.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.

    Example:
        >>> from setuptools import setup
        >>> from torch.utils.cpp_extension import BuildExtension, CppExtension
        >>> setup(
                name='cuda_extension',
                ext_modules=[
                    CUDAExtension(
                            name='cuda_extension',
                            sources=['extension.cpp', 'extension_kernel.cu'],
                            extra_compile_args={'cxx': ['-g'],
                                                'nvcc': ['-O2']})
                ],
                cmdclass={
                    'build_ext': BuildExtension
                })
    '''
    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += library_paths(cuda=True)
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('cudart')
    if sys.platform == 'win32':
        libraries.append('ATen')
        libraries.append('_C')
    kwargs['libraries'] = libraries

    include_dirs = kwargs.get('include_dirs', [])
    include_dirs += include_paths(cuda=True)
    kwargs['include_dirs'] = include_dirs

    kwargs['language'] = 'c++'

    return setuptools.Extension(name, sources, *args, **kwargs)


def include_paths(cuda=False):
    '''
    Get the include paths required to build a C++ or CUDA extension.

    Args:
        cuda: If `True`, includes CUDA-specific include paths.

    Returns:
        A list of include path strings.
    '''
    here = os.path.abspath(__file__)
    torch_path = os.path.dirname(os.path.dirname(here))
    lib_include = os.path.join(torch_path, 'lib', 'include')
    # Some internal (old) Torch headers don't properly prefix their includes,
    # so we need to pass -Itorch/lib/include/TH as well.
    paths = [
        lib_include,
        os.path.join(lib_include, 'TH'),
        os.path.join(lib_include, 'THC')
    ]
    if cuda:
        paths.append(_join_cuda_home('include'))
    return paths


def library_paths(cuda=False):
    '''
    Get the library paths required to build a C++ or CUDA extension.

    Args:
        cuda: If `True`, includes CUDA-specific library paths.

    Returns:
        A list of library path strings.
    '''
    paths = []

    if sys.platform == 'win32':
        here = os.path.abspath(__file__)
        torch_path = os.path.dirname(os.path.dirname(here))
        lib_path = os.path.join(torch_path, 'lib')

        paths.append(lib_path)

    if cuda:
        lib_dir = 'lib/x64' if sys.platform == 'win32' else 'lib64'
        paths.append(_join_cuda_home(lib_dir))
    return paths


def load(name,
         sources,
         extra_cflags=None,
         extra_cuda_cflags=None,
         extra_ldflags=None,
         extra_include_paths=None,
         build_directory=None,
         verbose=False):
    '''
    Loads a PyTorch C++ extension just-in-time (JIT).

    To load an extension, a Ninja build file is emitted, which is used to
    compile the given sources into a dynamic library. This library is
    subsequently loaded into the current Python process as a module and
    returned from this function, ready for use.

    By default, the directory to which the build file is emitted and the
    resulting library compiled to is ``<tmp>/torch_extensions/<name>``, where
    ``<tmp>`` is the temporary folder on the current platform and ``<name>``
    the name of the extension. This location can be overridden in two ways.
    First, if the ``TORCH_EXTENSIONS_DIR`` environment variable is set, it
    replaces ``<tmp>/torch_extensions`` and all extensions will be compiled
    into subfolders of this directory. Second, if the ``build_directory``
    argument to this function is supplied, it overrides the entire path, i.e.
    the library will be compiled into that folder directly.

    To compile the sources, the default system compiler (``c++``) is used,
    which can be overridden by setting the ``CXX`` environment variable. To pass
    additional arguments to the compilation process, ``extra_cflags`` or
    ``extra_ldflags`` can be provided. For example, to compile your extension
    with optimizations, pass ``extra_cflags=['-O3']``. You can also use
    ``extra_cflags`` to pass further include directories.

    CUDA support with mixed compilation is provided. Simply pass CUDA source
    files (``.cu`` or ``.cuh``) along with other sources. Such files will be
    detected and compiled with nvcc rather than the C++ compiler. This includes
    passing the CUDA lib64 directory as a library directory, and linking
    ``cudart``. You can pass additional flags to nvcc via
    ``extra_cuda_cflags``, just like with ``extra_cflags`` for C++. Various
    heuristics for finding the CUDA install directory are used, which usually
    work fine. If not, setting the ``CUDA_HOME`` environment variable is the
    safest option.

    Args:
        name: The name of the extension to build. This MUST be the same as the
            name of the pybind11 module!
        sources: A list of relative or absolute paths to C++ source files.
        extra_cflags: optional list of compiler flags to forward to the build.
        extra_cuda_cflags: optional list of compiler flags to forward to nvcc
            when building CUDA sources.
        extra_ldflags: optional list of linker flags to forward to the build.
        extra_include_paths: optional list of include directories to forward
            to the build.
        build_directory: optional path to use as build workspace.
        verbose: If ``True``, turns on verbose logging of load steps.

    Returns:
        The loaded PyTorch extension as a Python module.

    Example:
        >>> from torch.utils.cpp_extension import load
        >>> module = load(
                name='extension',
                sources=['extension.cpp', 'extension_kernel.cu'],
                extra_cflags=['-O2'],
                verbose=True)
    '''

    verify_ninja_availability()

    # Allows sources to be a single path or a list of paths.
    if isinstance(sources, str):
        sources = [sources]

    if build_directory is None:
        build_directory = _get_build_directory(name, verbose)

    baton = FileBaton(os.path.join(build_directory, 'lock'))

    if baton.try_acquire():
        try:
            with_cuda = any(map(_is_cuda_file, sources))
            extra_ldflags = _prepare_ldflags(
                extra_ldflags or [],
                with_cuda,
                verbose)
            build_file_path = os.path.join(build_directory, 'build.ninja')
            if verbose:
                print(
                    'Emitting ninja build file {}...'.format(build_file_path))
            # NOTE: Emitting a new ninja build file does not cause re-compilation if
            # the sources did not change, so it's ok to re-emit (and it's fast).
            _write_ninja_file(
                path=build_file_path,
                name=name,
                sources=sources,
                extra_cflags=extra_cflags or [],
                extra_cuda_cflags=extra_cuda_cflags or [],
                extra_ldflags=extra_ldflags or [],
                extra_include_paths=extra_include_paths or [],
                with_cuda=with_cuda)

            if verbose:
                print('Building extension module {}...'.format(name))
            _build_extension_module(name, build_directory)
        finally:
            baton.release()
    else:
        baton.wait()

    if verbose:
        print('Loading extension module {}...'.format(name))
    return _import_module_from_library(name, build_directory)


def verify_ninja_availability():
    '''
    Returns ``True`` if the `ninja <https://ninja-build.org/>`_ build system is
    available on the system.
    '''
    with open(os.devnull, 'wb') as devnull:
        try:
            subprocess.check_call('ninja --version'.split(), stdout=devnull)
        except OSError:
            raise RuntimeError("Ninja is required to load C++ extensions")


def _prepare_ldflags(extra_ldflags, with_cuda, verbose):
    if sys.platform == 'win32':
        python_path = os.path.dirname(sys.executable)
        python_lib_path = os.path.join(python_path, 'libs')

        here = os.path.abspath(__file__)
        torch_path = os.path.dirname(os.path.dirname(here))
        lib_path = os.path.join(torch_path, 'lib')

        extra_ldflags.append('ATen.lib')
        extra_ldflags.append('_C.lib')
        extra_ldflags.append('/LIBPATH:{}'.format(python_lib_path))
        extra_ldflags.append('/LIBPATH:{}'.format(lib_path))

    if with_cuda:
        if verbose:
            print('Detected CUDA files, patching ldflags')
        if sys.platform == 'win32':
            extra_ldflags.append('/LIBPATH:{}'.format(
                _join_cuda_home('lib/x64')))
            extra_ldflags.append('cudart.lib')
        else:
            extra_ldflags.append('-L{}'.format(_join_cuda_home('lib64')))
            extra_ldflags.append('-lcudart')

    return extra_ldflags


def _get_build_directory(name, verbose):
    root_extensions_directory = os.environ.get('TORCH_EXTENSIONS_DIR')
    if root_extensions_directory is None:
        # tempfile.gettempdir() will be /tmp on UNIX and \TEMP on Windows.
        root_extensions_directory = os.path.join(tempfile.gettempdir(),
                                                 'torch_extensions')

    if verbose:
        print('Using {} as PyTorch extensions root...'.format(
            root_extensions_directory))

    build_directory = os.path.join(root_extensions_directory, name)
    if not os.path.exists(build_directory):
        if verbose:
            print('Creating extension directory {}...'.format(build_directory))
        # This is like mkdir -p, i.e. will also create parent directories.
        os.makedirs(build_directory)

    return build_directory


def _build_extension_module(name, build_directory):
    try:
        subprocess.check_output(
            ['ninja', '-v'], stderr=subprocess.STDOUT, cwd=build_directory)
    except subprocess.CalledProcessError:
        # Python 2 and 3 compatible way of getting the error object.
        _, error, _ = sys.exc_info()
        # error.output contains the stdout and stderr of the build attempt.
        raise RuntimeError("Error building extension '{}': {}".format(
            name, error.output.decode()))


def _import_module_from_library(module_name, path):
    # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
    file, path, description = imp.find_module(module_name, [path])
    # Close the .so file after load.
    with file:
        return imp.load_module(module_name, file, path, description)


def _write_ninja_file(path,
                      name,
                      sources,
                      extra_cflags,
                      extra_cuda_cflags,
                      extra_ldflags,
                      extra_include_paths,
                      with_cuda=False):
    # Version 1.3 is required for the `deps` directive.
    config = ['ninja_required_version = 1.3']
    config.append('cxx = {}'.format(os.environ.get('CXX', 'c++')))
    if with_cuda:
        config.append('nvcc = {}'.format(_join_cuda_home('bin', 'nvcc')))

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [os.path.abspath(file) for file in sources]
    includes = [os.path.abspath(file) for file in extra_include_paths]

    # include_paths() gives us the location of torch/torch.h
    includes += include_paths(with_cuda)
    # sysconfig.get_paths()['include'] gives us the location of Python.h
    includes.append(sysconfig.get_paths()['include'])

    common_cflags = ['-DTORCH_EXTENSION_NAME={}'.format(name)]
    common_cflags += ['-I{}'.format(include) for include in includes]

    cflags = common_cflags + ['-fPIC', '-std=c++11'] + extra_cflags
    if sys.platform == 'win32':
        from distutils.spawn import _nt_quote_args
        cflags = _nt_quote_args(cflags)
    flags = ['cflags = {}'.format(' '.join(cflags))]

    if with_cuda:
        cuda_flags = common_cflags
        if sys.platform == 'win32':
            cuda_flags = _nt_quote_args(cuda_flags)
        else:
            cuda_flags += ['--compiler-options', "'-fPIC'"]
            cuda_flags += extra_cuda_cflags
            if not any(flag.startswith('-std=') for flag in cuda_flags):
                cuda_flags.append('-std=c++11')

        flags.append('cuda_flags = {}'.format(' '.join(cuda_flags)))

    if sys.platform == 'win32':
        ldflags = ['/DLL'] + extra_ldflags
    else:
        ldflags = ['-shared'] + extra_ldflags
    # The darwin linker needs explicit consent to ignore unresolved symbols.
    if sys.platform == 'darwin':
        ldflags.append('-undefined dynamic_lookup')
    elif sys.platform == 'win32':
        ldflags = _nt_quote_args(ldflags)
    flags.append('ldflags = {}'.format(' '.join(ldflags)))

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ['rule compile']
    if sys.platform == 'win32':
        compile_rule.append(
            '  command = cl /showIncludes $cflags -c $in /Fo$out')
        compile_rule.append('  deps = msvc')
    else:
        compile_rule.append(
            '  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out')
        compile_rule.append('  depfile = $out.d')
        compile_rule.append('  deps = gcc')

    if with_cuda:
        cuda_compile_rule = ['rule cuda_compile']
        cuda_compile_rule.append(
            '  command = $nvcc $cuda_flags -c $in -o $out')

    link_rule = ['rule link']
    if sys.platform == 'win32':
        cl_paths = subprocess.check_output(['where',
                                            'cl']).decode().split('\r\n')
        if len(cl_paths) >= 1:
            cl_path = os.path.dirname(cl_paths[0]).replace(':', '$:')
        else:
            raise RuntimeError("MSVC is required to load C++ extensions")
        link_rule.append(
            '  command = "{}/link.exe" $in /nologo $ldflags /out:$out'.format(
                cl_path))
    else:
        link_rule.append('  command = $cxx $ldflags $in -o $out')

    # Emit one build rule per source to enable incremental build.
    object_files = []
    build = []
    for source_file in sources:
        # '/path/to/file.cpp' -> 'file'
        file_name = os.path.splitext(os.path.basename(source_file))[0]
        if _is_cuda_file(source_file):
            rule = 'cuda_compile'
            # Use a different object filename in case a C++ and CUDA file have
            # the same filename but different extension (.cpp vs. .cu).
            target = '{}.cuda.o'.format(file_name)
        else:
            rule = 'compile'
            target = '{}.o'.format(file_name)
        object_files.append(target)
        if sys.platform == 'win32':
            source_file = source_file.replace(':', '$:')
        build.append('build {}: {} {}'.format(target, rule, source_file))

    ext = '.pyd' if sys.platform == 'win32' else '.so'
    library_target = '{}{}'.format(name, ext)
    link = ['build {}: link {}'.format(library_target, ' '.join(object_files))]

    default = ['default {}'.format(library_target)]

    # 'Blocks' should be separated by newlines, for visual benefit.
    blocks = [config, flags, compile_rule]
    if with_cuda:
        blocks.append(cuda_compile_rule)
    blocks += [link_rule, build, link, default]
    with open(path, 'w') as build_file:
        for block in blocks:
            lines = '\n'.join(block)
            build_file.write('{}\n\n'.format(lines))


def _join_cuda_home(*paths):
    '''
    Joins paths with CUDA_HOME, or raises an error if it CUDA_HOME is not set.

    This is basically a lazy way of raising an error for missing $CUDA_HOME
    only once we need to get any CUDA-specific path.
    '''
    if CUDA_HOME is None:
        raise EnvironmentError('CUDA_HOME environment variable is not set. '
                               'Please set it to your CUDA install root.')
    return os.path.join(CUDA_HOME, *paths)


def _is_cuda_file(path):
    return os.path.splitext(path)[1] in ['.cu', '.cuh']

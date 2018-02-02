import imp
import os
import re
import subprocess
import sys
import sysconfig
import tempfile
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


def load(name,
         sources,
         extra_cflags=None,
         extra_ldflags=None,
         extra_include_paths=None,
         build_directory=None,
         verbose=False):
    '''
    Loads a C++ PyTorch extension.

    To load an extension, a Ninja build file is emitted, which is used to
    compile the given sources into a dynamic library. This library is
    subsequently loaded into the current Python process as a module and
    returned from this function, ready for use.

    By default, the directory to which the build file is emitted and the
    resulting library compiled to is `<tmp>/torch_extensions`, where `<tmp>` is
    the temporary folder on the current platform. This location can be
    overriden in two ways. First, if the `TORCH_EXTENSIONS_DIR` environment
    variable is set, it replaces `<tmp>` and all extensions will be compiled
    into subfolders of this directory. Second, if the `build_directory`
    argument to this function is supplied, it overrides the entire path, i.e.
    the library will be compiled into that folder directly.

    To compile the sources, the default system compiler (`c++`) is used, which
    can be overriden by setting the CXX environment variable. To pass
    additional arguments to the compilation process, `extra_cflags` or
    `extra_ldflags` can be provided. For example, to compile your extension
    with optimizations, pass `extra_cflags=['-O3']`. You can also use
    `extra_cflags` to pass further include directories (`-I`).

    Args:
        name: The name of the module to build.
        sources: A list of relative or absolute paths to C++ source files.
        extra_cflags: optional list of compiler flags to forward to the build.
        extra_ldflags: optional list of linker flags to forward to the build.
        extra_include_paths: optional list of include directories to forward
            to the build.
        build_directory: optional path to use as build workspace.
        verbose: If `True`, turns on verbose logging of load steps.

    Returns:
        The loaded PyTorch extension as a Python module.
    '''

    verify_ninja_availability()

    # Allows sources to be a single path or a list of paths.
    if isinstance(sources, str):
        sources = [sources]

    if build_directory is None:
        build_directory = _get_build_directory(name, verbose)

    build_file_path = os.path.join(build_directory, 'build.ninja')
    if verbose:
        print('Emitting ninja build file {}...'.format(build_file_path))
    # NOTE: Emitting a new ninja build file does not cause re-compilation if
    # the sources did not change, so it's ok to re-emit (and it's fast).
    _write_ninja_file(build_file_path, name, sources, extra_cflags or [],
                      extra_ldflags or [], extra_include_paths or [])

    if verbose:
        print('Building extension module {}...'.format(name))
    _build_extension_module(name, build_directory)

    if verbose:
        print('Loading extension module {}...'.format(name))
    return _import_module_from_library(name, build_directory)


def verify_ninja_availability():
    with open(os.devnull, 'wb') as devnull:
        try:
            subprocess.check_call('ninja --version'.split(), stdout=devnull)
        except OSError:
            raise RuntimeError("Ninja is required to load C++ extensions")


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


def _write_ninja_file(path, name, sources, extra_cflags, extra_ldflags,
                      extra_include_paths):
    # Version 1.3 is required for the `deps` directive.
    config = ['ninja_required_version = 1.3']
    config.append('cxx = {}'.format(os.environ.get('CXX', 'c++')))

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [os.path.abspath(file) for file in sources]
    includes = [os.path.abspath(file) for file in extra_include_paths]

    # include_paths() gives us the location of torch/torch.h
    includes += include_paths()
    # sysconfig.get_paths()['include'] gives us the location of Python.h
    includes.append(sysconfig.get_paths()['include'])

    cflags = ['-fPIC', '-std=c++11']
    cflags += ['-I{}'.format(include) for include in includes]
    cflags += extra_cflags
    flags = ['cflags = {}'.format(' '.join(cflags))]

    ldflags = ['-shared'] + extra_ldflags
    # The darwin linker needs explicit consent to ignore unresolved symbols
    if sys.platform == 'darwin':
        ldflags.append('-undefined dynamic_lookup')
    flags.append('ldflags = {}'.format(' '.join(ldflags)))

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ['rule compile']
    compile_rule.append(
        '  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out')
    compile_rule.append('  depfile = $out.d')
    compile_rule.append('  deps = gcc')
    compile_rule.append('')

    link_rule = ['rule link']
    link_rule.append('  command = $cxx $ldflags $in -o $out')

    # Emit one build rule per source to enable incremental build.
    object_files = []
    build = []
    for source_file in sources:
        # '/path/to/file.cpp' -> 'file'
        file_name = os.path.splitext(os.path.basename(source_file))[0]
        target = '{}.o'.format(file_name)
        object_files.append(target)
        build.append('build {}: compile {}'.format(target, source_file))

    library_target = '{}.so'.format(name)
    link = ['build {}: link {}'.format(library_target, ' '.join(object_files))]

    default = ['default {}'.format(library_target)]

    # 'Blocks' should be separated by newlines, for visual benefit.
    blocks = [config, flags, compile_rule, link_rule, build, link, default]
    with open(path, 'w') as build_file:
        for block in blocks:
            lines = '\n'.join(block)
            build_file.write('{}\n\n'.format(lines))

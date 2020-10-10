import sys
import subprocess
import pkgutil
import types
import importlib
import warnings

import numpy as np
import numpy
import pytest

try:
    import ctypes
except ImportError:
    ctypes = None


def check_dir(module, module_name=None):
    """Returns a mapping of all objects with the wrong __module__ attribute."""
    if module_name is None:
        module_name = module.__name__
    results = {}
    for name in dir(module):
        item = getattr(module, name)
        if (hasattr(item, '__module__') and hasattr(item, '__name__')
                and item.__module__ != module_name):
            results[name] = item.__module__ + '.' + item.__name__
    return results


def test_numpy_namespace():
    # None of these objects are publicly documented to be part of the main
    # NumPy namespace (some are useful though, others need to be cleaned up)
    undocumented = {
        'Tester': 'numpy.testing._private.nosetester.NoseTester',
        '_add_newdoc_ufunc': 'numpy.core._multiarray_umath._add_newdoc_ufunc',
        'add_docstring': 'numpy.core._multiarray_umath.add_docstring',
        'add_newdoc': 'numpy.core.function_base.add_newdoc',
        'add_newdoc_ufunc': 'numpy.core._multiarray_umath._add_newdoc_ufunc',
        'byte_bounds': 'numpy.lib.utils.byte_bounds',
        'compare_chararrays': 'numpy.core._multiarray_umath.compare_chararrays',
        'deprecate': 'numpy.lib.utils.deprecate',
        'deprecate_with_doc': 'numpy.lib.utils.<lambda>',
        'disp': 'numpy.lib.function_base.disp',
        'fastCopyAndTranspose': 'numpy.core._multiarray_umath._fastCopyAndTranspose',
        'get_array_wrap': 'numpy.lib.shape_base.get_array_wrap',
        'get_include': 'numpy.lib.utils.get_include',
        'mafromtxt': 'numpy.lib.npyio.mafromtxt',
        'ndfromtxt': 'numpy.lib.npyio.ndfromtxt',
        'recfromcsv': 'numpy.lib.npyio.recfromcsv',
        'recfromtxt': 'numpy.lib.npyio.recfromtxt',
        'safe_eval': 'numpy.lib.utils.safe_eval',
        'set_string_function': 'numpy.core.arrayprint.set_string_function',
        'show_config': 'numpy.__config__.show',
        'who': 'numpy.lib.utils.who',
    }
    # These built-in types are re-exported by numpy.
    builtins = {
        'bool': 'builtins.bool',
        'complex': 'builtins.complex',
        'float': 'builtins.float',
        'int': 'builtins.int',
        'long': 'builtins.int',
        'object': 'builtins.object',
        'str': 'builtins.str',
        'unicode': 'builtins.str',
    }
    whitelist = dict(undocumented, **builtins)
    bad_results = check_dir(np)
    # pytest gives better error messages with the builtin assert than with
    # assert_equal
    assert bad_results == whitelist


@pytest.mark.parametrize('name', ['testing', 'Tester'])
def test_import_lazy_import(name):
    """Make sure we can actually use the modules we lazy load.

    While not exported as part of the public API, it was accessible.  With the
    use of __getattr__ and __dir__, this isn't always true It can happen that
    an infinite recursion may happen.

    This is the only way I found that would force the failure to appear on the
    badly implemented code.

    We also test for the presence of the lazily imported modules in dir

    """
    exe = (sys.executable, '-c', "import numpy; numpy." + name)
    result = subprocess.check_output(exe)
    assert not result

    # Make sure they are still in the __dir__
    assert name in dir(np)


def test_dir_testing():
    """Assert that output of dir has only one "testing/tester"
    attribute without duplicate"""
    assert len(dir(np)) == len(set(dir(np)))
    

def test_numpy_linalg():
    bad_results = check_dir(np.linalg)
    assert bad_results == {}


def test_numpy_fft():
    bad_results = check_dir(np.fft)
    assert bad_results == {}


@pytest.mark.skipif(ctypes is None,
                    reason="ctypes not available in this python")
def test_NPY_NO_EXPORT():
    cdll = ctypes.CDLL(np.core._multiarray_tests.__file__)
    # Make sure an arbitrary NPY_NO_EXPORT function is actually hidden
    f = getattr(cdll, 'test_not_exported', None)
    assert f is None, ("'test_not_exported' is mistakenly exported, "
                      "NPY_NO_EXPORT does not work")


# Historically NumPy has not used leading underscores for private submodules
# much.  This has resulted in lots of things that look like public modules
# (i.e. things that can be imported as `import numpy.somesubmodule.somefile`),
# but were never intended to be public.  The PUBLIC_MODULES list contains
# modules that are either public because they were meant to be, or because they
# contain public functions/objects that aren't present in any other namespace
# for whatever reason and therefore should be treated as public.
#
# The PRIVATE_BUT_PRESENT_MODULES list contains modules that look public (lack
# of underscores) but should not be used.  For many of those modules the
# current status is fine.  For others it may make sense to work on making them
# private, to clean up our public API and avoid confusion.
PUBLIC_MODULES = ['numpy.' + s for s in [
    "ctypeslib",
    "distutils",
    "distutils.cpuinfo",
    "distutils.exec_command",
    "distutils.misc_util",
    "distutils.log",
    "distutils.system_info",
    "doc",
    "doc.basics",
    "doc.broadcasting",
    "doc.byteswapping",
    "doc.constants",
    "doc.creation",
    "doc.dispatch",
    "doc.glossary",
    "doc.indexing",
    "doc.internals",
    "doc.misc",
    "doc.structured_arrays",
    "doc.subclassing",
    "doc.ufuncs",
    "dual",
    "f2py",
    "fft",
    "lib",
    "lib.format",  # was this meant to be public?
    "lib.mixins",
    "lib.recfunctions",
    "lib.scimath",
    "linalg",
    "ma",
    "ma.extras",
    "ma.mrecords",
    "matlib",
    "polynomial",
    "polynomial.chebyshev",
    "polynomial.hermite",
    "polynomial.hermite_e",
    "polynomial.laguerre",
    "polynomial.legendre",
    "polynomial.polynomial",
    "polynomial.polyutils",
    "random",
    "testing",
    "version",
]]


PUBLIC_ALIASED_MODULES = [
    "numpy.char",
    "numpy.emath",
    "numpy.rec",
]


PRIVATE_BUT_PRESENT_MODULES = ['numpy.' + s for s in [
    "compat",
    "compat.py3k",
    "conftest",
    "core",
    "core.arrayprint",
    "core.defchararray",
    "core.einsumfunc",
    "core.fromnumeric",
    "core.function_base",
    "core.getlimits",
    "core.machar",
    "core.memmap",
    "core.multiarray",
    "core.numeric",
    "core.numerictypes",
    "core.overrides",
    "core.records",
    "core.shape_base",
    "core.umath",
    "core.umath_tests",
    "distutils.ccompiler",
    "distutils.command",
    "distutils.command.autodist",
    "distutils.command.bdist_rpm",
    "distutils.command.build",
    "distutils.command.build_clib",
    "distutils.command.build_ext",
    "distutils.command.build_py",
    "distutils.command.build_scripts",
    "distutils.command.build_src",
    "distutils.command.config",
    "distutils.command.config_compiler",
    "distutils.command.develop",
    "distutils.command.egg_info",
    "distutils.command.install",
    "distutils.command.install_clib",
    "distutils.command.install_data",
    "distutils.command.install_headers",
    "distutils.command.sdist",
    "distutils.conv_template",
    "distutils.core",
    "distutils.extension",
    "distutils.fcompiler",
    "distutils.fcompiler.absoft",
    "distutils.fcompiler.compaq",
    "distutils.fcompiler.environment",
    "distutils.fcompiler.g95",
    "distutils.fcompiler.gnu",
    "distutils.fcompiler.hpux",
    "distutils.fcompiler.ibm",
    "distutils.fcompiler.intel",
    "distutils.fcompiler.lahey",
    "distutils.fcompiler.mips",
    "distutils.fcompiler.nag",
    "distutils.fcompiler.none",
    "distutils.fcompiler.pathf95",
    "distutils.fcompiler.pg",
    "distutils.fcompiler.sun",
    "distutils.fcompiler.vast",
    "distutils.from_template",
    "distutils.intelccompiler",
    "distutils.lib2def",
    "distutils.line_endings",
    "distutils.mingw32ccompiler",
    "distutils.msvccompiler",
    "distutils.npy_pkg_config",
    "distutils.numpy_distribution",
    "distutils.pathccompiler",
    "distutils.unixccompiler",
    "f2py.auxfuncs",
    "f2py.capi_maps",
    "f2py.cb_rules",
    "f2py.cfuncs",
    "f2py.common_rules",
    "f2py.crackfortran",
    "f2py.diagnose",
    "f2py.f2py2e",
    "f2py.f2py_testing",
    "f2py.f90mod_rules",
    "f2py.func2subr",
    "f2py.rules",
    "f2py.use_rules",
    "fft.helper",
    "lib.arraypad",
    "lib.arraysetops",
    "lib.arrayterator",
    "lib.financial",
    "lib.function_base",
    "lib.histograms",
    "lib.index_tricks",
    "lib.nanfunctions",
    "lib.npyio",
    "lib.polynomial",
    "lib.shape_base",
    "lib.stride_tricks",
    "lib.twodim_base",
    "lib.type_check",
    "lib.ufunclike",
    "lib.user_array",  # note: not in np.lib, but probably should just be deleted
    "lib.utils",
    "linalg.lapack_lite",
    "linalg.linalg",
    "ma.bench",
    "ma.core",
    "ma.testutils",
    "ma.timer_comparison",
    "matrixlib",
    "matrixlib.defmatrix",
    "random.mtrand",
    "random.bit_generator",
    "testing.print_coercion_tables",
    "testing.utils",
]]


def is_unexpected(name):
    """Check if this needs to be considered."""
    if '._' in name or '.tests' in name or '.setup' in name:
        return False

    if name in PUBLIC_MODULES:
        return False

    if name in PUBLIC_ALIASED_MODULES:
        return False

    if name in PRIVATE_BUT_PRESENT_MODULES:
        return False

    return True


# These are present in a directory with an __init__.py but cannot be imported
# code_generators/ isn't installed, but present for an inplace build
SKIP_LIST = [
    "numpy.core.code_generators",
    "numpy.core.code_generators.genapi",
    "numpy.core.code_generators.generate_umath",
    "numpy.core.code_generators.ufunc_docstrings",
    "numpy.core.code_generators.generate_numpy_api",
    "numpy.core.code_generators.generate_ufunc_api",
    "numpy.core.code_generators.numpy_api",
    "numpy.core.cversions",
    "numpy.core.generate_numpy_api",
    "numpy.distutils.msvc9compiler",
]


def test_all_modules_are_expected():
    """
    Test that we don't add anything that looks like a new public module by
    accident.  Check is based on filenames.
    """

    modnames = []
    for _, modname, ispkg in pkgutil.walk_packages(path=np.__path__,
                                                   prefix=np.__name__ + '.',
                                                   onerror=None):
        if is_unexpected(modname) and modname not in SKIP_LIST:
            # We have a name that is new.  If that's on purpose, add it to
            # PUBLIC_MODULES.  We don't expect to have to add anything to
            # PRIVATE_BUT_PRESENT_MODULES.  Use an underscore in the name!
            modnames.append(modname)

    if modnames:
        raise AssertionError("Found unexpected modules: {}".format(modnames))


# Stuff that clearly shouldn't be in the API and is detected by the next test
# below
SKIP_LIST_2 = [
    'numpy.math',
    'numpy.distutils.log.sys',
    'numpy.distutils.system_info.copy',
    'numpy.distutils.system_info.distutils',
    'numpy.distutils.system_info.log',
    'numpy.distutils.system_info.os',
    'numpy.distutils.system_info.platform',
    'numpy.distutils.system_info.re',
    'numpy.distutils.system_info.shutil',
    'numpy.distutils.system_info.subprocess',
    'numpy.distutils.system_info.sys',
    'numpy.distutils.system_info.tempfile',
    'numpy.distutils.system_info.textwrap',
    'numpy.distutils.system_info.warnings',
    'numpy.doc.constants.re',
    'numpy.doc.constants.textwrap',
    'numpy.lib.emath',
    'numpy.lib.math',
    'numpy.matlib.char',
    'numpy.matlib.rec',
    'numpy.matlib.emath',
    'numpy.matlib.math',
    'numpy.matlib.linalg',
    'numpy.matlib.fft',
    'numpy.matlib.random',
    'numpy.matlib.ctypeslib',
    'numpy.matlib.ma',
]


def test_all_modules_are_expected_2():
    """
    Method checking all objects. The pkgutil-based method in
    `test_all_modules_are_expected` does not catch imports into a namespace,
    only filenames.  So this test is more thorough, and checks this like:

        import .lib.scimath as emath

    To check if something in a module is (effectively) public, one can check if
    there's anything in that namespace that's a public function/object but is
    not exposed in a higher-level namespace.  For example for a `numpy.lib`
    submodule::

        mod = np.lib.mixins
        for obj in mod.__all__:
            if obj in np.__all__:
                continue
            elif obj in np.lib.__all__:
                continue

            else:
                print(obj)

    """

    def find_unexpected_members(mod_name):
        members = []
        module = importlib.import_module(mod_name)
        if hasattr(module, '__all__'):
            objnames = module.__all__
        else:
            objnames = dir(module)

        for objname in objnames:
            if not objname.startswith('_'):
                fullobjname = mod_name + '.' + objname
                if isinstance(getattr(module, objname), types.ModuleType):
                    if is_unexpected(fullobjname):
                        if fullobjname not in SKIP_LIST_2:
                            members.append(fullobjname)

        return members

    unexpected_members = find_unexpected_members("numpy")
    for modname in PUBLIC_MODULES:
        unexpected_members.extend(find_unexpected_members(modname))

    if unexpected_members:
        raise AssertionError("Found unexpected object(s) that look like "
                             "modules: {}".format(unexpected_members))


def test_api_importable():
    """
    Check that all submodules listed higher up in this file can be imported

    Note that if a PRIVATE_BUT_PRESENT_MODULES entry goes missing, it may
    simply need to be removed from the list (deprecation may or may not be
    needed - apply common sense).
    """
    def check_importable(module_name):
        try:
            importlib.import_module(module_name)
        except (ImportError, AttributeError):
            return False

        return True

    module_names = []
    for module_name in PUBLIC_MODULES:
        if not check_importable(module_name):
            module_names.append(module_name)

    if module_names:
        raise AssertionError("Modules in the public API that cannot be "
                             "imported: {}".format(module_names))

    for module_name in PUBLIC_ALIASED_MODULES:
        try:
            eval(module_name)
        except AttributeError:
            module_names.append(module_name)

    if module_names:
        raise AssertionError("Modules in the public API that were not "
                             "found: {}".format(module_names))

    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', category=DeprecationWarning)
        warnings.filterwarnings('always', category=ImportWarning)
        for module_name in PRIVATE_BUT_PRESENT_MODULES:
            if not check_importable(module_name):
                module_names.append(module_name)

    if module_names:
        raise AssertionError("Modules that are not really public but looked "
                             "public and can not be imported: "
                             "{}".format(module_names))

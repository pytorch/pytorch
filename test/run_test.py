#!/usr/bin/env python

from __future__ import print_function

import argparse
from datetime import datetime
import modulefinder
import os
import shutil
import signal
import subprocess
import sys
import tempfile

import torch
import torch._six
from torch.utils import cpp_extension
from torch.testing._internal.common_utils import TEST_WITH_ROCM, shell
import torch.distributed as dist
PY2 = sys.version_info <= (3,)
PY33 = sys.version_info >= (3, 3)
PY36 = sys.version_info >= (3, 6)

TESTS = [
    'test_autograd',
    'test_complex',
    'test_cpp_api_parity',
    'test_cpp_extensions_aot_no_ninja',
    'test_cpp_extensions_aot_ninja',
    'test_cpp_extensions_jit',
    'distributed/test_c10d',
    'distributed/test_c10d_spawn',
    'test_cuda',
    'test_jit_cuda_fuser',
    'test_cuda_primary_ctx',
    'test_dataloader',
    'distributed/test_data_parallel',
    'distributed/test_distributed',
    'test_distributions',
    'test_docs_coverage',
    'test_expecttest',
    'test_indexing',
    'test_jit',
    'test_logging',
    'test_mkldnn',
    'test_multiprocessing',
    'test_multiprocessing_spawn',
    'distributed/test_nccl',
    'test_nn',
    'test_numba_integration',
    'test_optim',
    'quantization/test_fake_quant',
    'quantization/test_numerics',
    'quantization/test_qat',
    'quantization/test_quantization',
    'quantization/test_numeric_suite',
    'quantization/test_quantized',
    'quantization/test_quantized_tensor',
    'quantization/test_quantized_nn_mods',
    'quantization/test_quantize_script',
    'test_sparse',
    'test_serialization',
    'test_show_pickle',
    'test_torch',
    'test_type_info',
    'test_type_hints',
    'test_utils',
    'test_namedtuple_return_api',
    'test_jit_fuser',
    'test_jit_simple',
    'test_jit_legacy',
    'test_jit_fuser_legacy',
    'test_tensorboard',
    'test_namedtensor',
    'test_type_promotion',
    'test_jit_disabled',
    'test_function_schema',
    'test_overrides',
    'test_jit_fuser_te',
    'test_tensorexpr',
]

# skip < 3.3 because mock is added in 3.3 and is used in rpc_spawn
# skip python2 for rpc and dist_autograd tests that do not support python2
if PY33:
    TESTS.extend([
        'distributed/rpc/faulty_agent/test_dist_autograd_spawn',
        'distributed/rpc/faulty_agent/test_rpc_spawn',
        'distributed/rpc/jit/test_dist_autograd_spawn',
        'distributed/rpc/test_dist_autograd_spawn',
        'distributed/rpc/test_dist_optimizer_spawn',
        'distributed/rpc/test_rpc_spawn',
    ])

# skip < 3.6 b/c fstrings added in 3.6
if PY36:
    TESTS.extend([
        'test_jit_py3',
        'test_determination',
        'distributed/rpc/jit/test_rpc_spawn',
        'distributed/rpc/faulty_agent/test_rpc_spawn',
    ])

WINDOWS_BLACKLIST = [
    'distributed/rpc/faulty_agent/test_dist_autograd_spawn',
    'distributed/rpc/faulty_agent/test_rpc_spawn',
    'distributed/rpc/jit/test_dist_autograd_spawn',
    'distributed/rpc/jit/test_rpc_spawn',
    'distributed/rpc/test_dist_autograd_spawn',
    'distributed/rpc/test_dist_optimizer_spawn',
    'distributed/rpc/test_rpc_spawn',
    'distributed/test_distributed',
]

ROCM_BLACKLIST = [
    'distributed/rpc/faulty_agent/test_dist_autograd_spawn',
    'distributed/rpc/faulty_agent/test_rpc_spawn',
    'distributed/rpc/jit/test_dist_autograd_spawn',
    'distributed/rpc/jit/test_rpc_spawn',
    'distributed/rpc/test_dist_autograd_spawn',
    'distributed/rpc/test_dist_optimizer_spawn',
    'distributed/rpc/test_rpc_spawn',
    'test_cpp_extensions_aot_ninja',
    'test_cpp_extensions_jit',
    'test_determination',
    'test_multiprocessing',
    'test_jit_simple',
    'test_jit_legacy',
    'test_jit_fuser_legacy',
    'test_tensorexpr',
]

# These tests are slow enough that it's worth calculating whether the patch
# touched any related files first.
SLOW_TESTS = [
    'test_nn',
    'test_autograd',
    'test_cpp_extensions_jit',
    'test_jit_legacy',
    'test_dataloader',
    'test_overrides',
    'test_jit_simple',
    'test_jit',
    'test_torch',
    'distributed/test_distributed',
    'distributed/rpc/test_rpc_spawn',
    'distributed/rpc/test_dist_autograd_spawn',
    'test_cuda',
    'test_cuda_primary_ctx',
    'test_cpp_extensions_aot_ninja',
    'test_cpp_extensions_aot_no_ninja',
    'test_serialization',
    'test_distributions',
    'test_optim',
    'test_utils',
    'test_multiprocessing',
    'test_tensorboard',
    'distributed/test_c10d',
    'distributed/test_c10d_spawn',
    'quantization/test_quantized',
    'quantization/test_quantization',
    'quantization/test_numeric_suite',
    'test_determination',
]
_DEP_MODULES_CACHE = {}

DISTRIBUTED_TESTS_CONFIG = {}


if dist.is_available():
    DISTRIBUTED_TESTS_CONFIG['test'] = {
        'WORLD_SIZE': '1'
    }
    if not TEST_WITH_ROCM and dist.is_mpi_available():
        DISTRIBUTED_TESTS_CONFIG['mpi'] = {
            'WORLD_SIZE': '3',
            'TEST_REPORT_SOURCE_OVERRIDE': 'dist-mpi'
        }
    if dist.is_nccl_available():
        DISTRIBUTED_TESTS_CONFIG['nccl'] = {
            'WORLD_SIZE': '2' if torch.cuda.device_count() == 2 else '3',
            'TEST_REPORT_SOURCE_OVERRIDE': 'dist-nccl'
        }
    if not TEST_WITH_ROCM and dist.is_gloo_available():
        DISTRIBUTED_TESTS_CONFIG['gloo'] = {
            'WORLD_SIZE': '2' if torch.cuda.device_count() == 2 else '3',
            'TEST_REPORT_SOURCE_OVERRIDE': 'dist-gloo'
        }

# https://stackoverflow.com/questions/2549939/get-signal-names-from-numbers-in-python
SIGNALS_TO_NAMES_DICT = {getattr(signal, n): n for n in dir(signal)
                         if n.startswith('SIG') and '_' not in n}

CPP_EXTENSIONS_ERROR = """
Ninja (https://ninja-build.org) is required for some of the C++ extensions
tests, but it could not be found. Install ninja with `pip install ninja`
or `conda install ninja`. Alternatively, disable said tests with
`run_test.py --exclude test_cpp_extensions_aot_ninja test_cpp_extensions_jit`.
"""


def print_to_stderr(message):
    print(message, file=sys.stderr)


def run_test(executable, test_module, test_directory, options, *extra_unittest_args):
    unittest_args = options.additional_unittest_args
    if options.verbose:
        unittest_args.append('--verbose')
    # Can't call `python -m unittest test_*` here because it doesn't run code
    # in `if __name__ == '__main__': `. So call `python test_*.py` instead.
    argv = [test_module + '.py'] + unittest_args + list(extra_unittest_args)

    command = executable + argv
    return shell(command, test_directory)


def test_cuda_primary_ctx(executable, test_module, test_directory, options):
    return run_test(executable, test_module, test_directory, options, '--subprocess')


def _test_cpp_extensions_aot(executable, test_module, test_directory, options, use_ninja):
    if use_ninja:
        try:
            cpp_extension.verify_ninja_availability()
        except RuntimeError:
            print(CPP_EXTENSIONS_ERROR)
            return 1

    # Wipe the build folder, if it exists already
    cpp_extensions_test_dir = os.path.join(test_directory, 'cpp_extensions')
    cpp_extensions_test_build_dir = os.path.join(cpp_extensions_test_dir, 'build')
    if os.path.exists(cpp_extensions_test_build_dir):
        shutil.rmtree(cpp_extensions_test_build_dir)

    # Build the test cpp extensions modules
    shell_env = os.environ.copy()
    shell_env['USE_NINJA'] = str(1 if use_ninja else 0)
    cmd = [sys.executable, 'setup.py', 'install', '--root', './install']
    return_code = shell(cmd, cwd=cpp_extensions_test_dir, env=shell_env)
    if return_code != 0:
        return return_code
    if sys.platform != 'win32':
        return_code = shell(cmd,
                            cwd=os.path.join(cpp_extensions_test_dir, 'no_python_abi_suffix_test'),
                            env=shell_env)
        if return_code != 0:
            return return_code

    # "install" the test modules and run tests
    python_path = os.environ.get('PYTHONPATH', '')
    try:
        cpp_extensions = os.path.join(test_directory, 'cpp_extensions')
        install_directory = ''
        # install directory is the one that is named site-packages
        for root, directories, _ in os.walk(os.path.join(cpp_extensions, 'install')):
            for directory in directories:
                if '-packages' in directory:
                    install_directory = os.path.join(root, directory)

        assert install_directory, 'install_directory must not be empty'
        os.environ['PYTHONPATH'] = os.pathsep.join([install_directory, python_path])
        return run_test(executable, test_module, test_directory, options)
    finally:
        os.environ['PYTHONPATH'] = python_path


def test_cpp_extensions_aot_ninja(executable, test_module, test_directory, options):
    return _test_cpp_extensions_aot(executable, 'test_cpp_extensions_aot', test_directory,
                                    options, use_ninja=True)


def test_cpp_extensions_aot_no_ninja(executable, test_module, test_directory, options):
    return _test_cpp_extensions_aot(executable, 'test_cpp_extensions_aot',
                                    test_directory, options, use_ninja=False)


def test_distributed(executable, test_module, test_directory, options):
    mpi_available = subprocess.call('command -v mpiexec', shell=True) == 0
    if options.verbose and not mpi_available:
        print_to_stderr(
            'MPI not available -- MPI backend tests will be skipped')
    config = DISTRIBUTED_TESTS_CONFIG
    for backend, env_vars in config.items():
        if backend == 'mpi' and not mpi_available:
            continue
        for with_init_file in {True, False}:
            tmp_dir = tempfile.mkdtemp()
            if options.verbose:
                with_init = ' with file init_method' if with_init_file else ''
                print_to_stderr(
                    'Running distributed tests for the {} backend{}'.format(
                        backend, with_init))
            os.environ['TEMP_DIR'] = tmp_dir
            os.environ['BACKEND'] = backend
            os.environ['INIT_METHOD'] = 'env://'
            os.environ.update(env_vars)
            if with_init_file:
                if test_module == "test_distributed":
                    init_method = 'file://{}/'.format(tmp_dir)
                else:
                    init_method = 'file://{}/shared_init_file'.format(tmp_dir)
                os.environ['INIT_METHOD'] = init_method
            try:
                os.mkdir(os.path.join(tmp_dir, 'barrier'))
                os.mkdir(os.path.join(tmp_dir, 'test_dir'))
                if backend == 'mpi':
                    # test mpiexec for --noprefix option
                    with open(os.devnull, 'w') as devnull:
                        noprefix_opt = '--noprefix' if subprocess.call(
                            'mpiexec -n 1 --noprefix bash -c ""', shell=True,
                            stdout=devnull, stderr=subprocess.STDOUT) == 0 else ''

                    mpiexec = ['mpiexec', '-n', '3', noprefix_opt] + executable

                    return_code = run_test(mpiexec, test_module,
                                           test_directory, options)
                else:
                    return_code = run_test(executable, test_module, test_directory,
                                           options)
                if return_code != 0:
                    return return_code
            finally:
                shutil.rmtree(tmp_dir)
    return 0


CUSTOM_HANDLERS = {
    'test_cuda_primary_ctx': test_cuda_primary_ctx,
    'test_cpp_extensions_aot_no_ninja': test_cpp_extensions_aot_no_ninja,
    'test_cpp_extensions_aot_ninja': test_cpp_extensions_aot_ninja,
    'distributed/test_distributed': test_distributed,
}


def parse_test_module(test):
    return test.split('.')[0]


class TestChoices(list):
    def __init__(self, *args, **kwargs):
        super(TestChoices, self).__init__(args[0])

    def __contains__(self, item):
        return list.__contains__(self, parse_test_module(item))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the PyTorch unit test suite',
        epilog='where TESTS is any of: {}'.format(', '.join(TESTS)))
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='print verbose information and test-by-test results')
    parser.add_argument(
        '--jit',
        '--jit',
        action='store_true',
        help='run all jit tests')
    parser.add_argument(
        '-pt', '--pytest', action='store_true',
        help='If true, use `pytest` to execute the tests. E.g., this runs '
             'TestTorch with pytest in verbose and coverage mode: '
             'python run_test.py -vci torch -pt')
    parser.add_argument(
        '-c', '--coverage', action='store_true', help='enable coverage')
    parser.add_argument(
        '-i',
        '--include',
        nargs='+',
        choices=TestChoices(TESTS),
        default=TESTS,
        metavar='TESTS',
        help='select a set of tests to include (defaults to ALL tests).'
             ' tests can be specified with module name, module.TestClass'
             ' or module.TestClass.test_method')
    parser.add_argument(
        '-x',
        '--exclude',
        nargs='+',
        choices=TESTS,
        metavar='TESTS',
        default=[],
        help='select a set of tests to exclude')
    parser.add_argument(
        '-f',
        '--first',
        choices=TESTS,
        metavar='TESTS',
        help='select the test to start from (excludes previous tests)')
    parser.add_argument(
        '-l',
        '--last',
        choices=TESTS,
        metavar='TESTS',
        help='select the last test to run (excludes following tests)')
    parser.add_argument(
        '--bring-to-front',
        nargs='+',
        choices=TestChoices(TESTS),
        default=[],
        metavar='TESTS',
        help='select a set of tests to run first. This can be used in situations'
             ' where you want to run all tests, but care more about some set, '
             'e.g. after making a change to a specific component')
    parser.add_argument(
        '--ignore-win-blacklist',
        action='store_true',
        help='always run blacklisted windows tests')
    parser.add_argument(
        '--determine-from',
        help='File of affected source filenames to determine which tests to run.')
    parser.add_argument(
        'additional_unittest_args',
        nargs='*',
        help='additional arguments passed through to unittest, e.g., '
             'python run_test.py -i sparse -- TestSparse.test_factory_size_check')
    return parser.parse_args()


def get_executable_command(options):
    if options.coverage:
        executable = ['coverage', 'run', '--parallel-mode', '--source torch']
    else:
        executable = [sys.executable]
    if options.pytest:
        executable += ['-m', 'pytest']
    return executable


def find_test_index(test, selected_tests, find_last_index=False):
    """Find the index of the first or last occurrence of a given test/test module in the list of selected tests.

    This function is used to determine the indices when slicing the list of selected tests when
    ``options.first``(:attr:`find_last_index`=False) and/or ``options.last``(:attr:`find_last_index`=True) are used.

    :attr:`selected_tests` can be a list that contains multiple consequent occurrences of tests
    as part of the same test module, e.g.:

    ```
    selected_tests = ['autograd', 'cuda', **'torch.TestTorch.test_acos',
                     'torch.TestTorch.test_tan', 'torch.TestTorch.test_add'**, 'utils']
    ```

    If :attr:`test`='torch' and :attr:`find_last_index`=False, result should be **2**.
    If :attr:`test`='torch' and :attr:`find_last_index`=True, result should be **4**.

    Arguments:
        test (str): Name of test to lookup
        selected_tests (list): List of tests
        find_last_index (bool, optional): should we lookup the index of first or last
            occurrence (first is default)

    Returns:
        index of the first or last occurrence of the given test
    """
    idx = 0
    found_idx = -1
    for t in selected_tests:
        if t.startswith(test):
            found_idx = idx
            if not find_last_index:
                break
        idx += 1
    return found_idx


def exclude_tests(exclude_list, selected_tests, exclude_message=None):
    for exclude_test in exclude_list:
        tests_copy = selected_tests[:]
        for test in tests_copy:
            if test.startswith(exclude_test):
                if exclude_message is not None:
                    print_to_stderr('Excluding {} {}'.format(test, exclude_message))
                selected_tests.remove(test)
    return selected_tests


def get_selected_tests(options):
    selected_tests = options.include

    if options.bring_to_front:
        to_front = set(options.bring_to_front)
        selected_tests = options.bring_to_front + list(filter(lambda name: name not in to_front,
                                                              selected_tests))

    if options.first:
        first_index = find_test_index(options.first, selected_tests)
        selected_tests = selected_tests[first_index:]

    if options.last:
        last_index = find_test_index(options.last, selected_tests, find_last_index=True)
        selected_tests = selected_tests[:last_index + 1]

    selected_tests = exclude_tests(options.exclude, selected_tests)

    if sys.platform == 'win32' and not options.ignore_win_blacklist:
        target_arch = os.environ.get('VSCMD_ARG_TGT_ARCH')
        if target_arch != 'x64':
            WINDOWS_BLACKLIST.append('cpp_extensions_aot_no_ninja')
            WINDOWS_BLACKLIST.append('cpp_extensions_aot_ninja')
            WINDOWS_BLACKLIST.append('cpp_extensions_jit')
            WINDOWS_BLACKLIST.append('jit')
            WINDOWS_BLACKLIST.append('jit_fuser')

        selected_tests = exclude_tests(WINDOWS_BLACKLIST, selected_tests, 'on Windows')

    elif TEST_WITH_ROCM:
        selected_tests = exclude_tests(ROCM_BLACKLIST, selected_tests, 'on ROCm')

    return selected_tests


def test_impact_of_file(filename):
    """Determine what class of impact this file has on test runs.

    Possible values:
        TORCH - torch python code
        CAFFE2 - caffe2 python code
        TEST - torch test code
        UNKNOWN - may affect all tests
        NONE - known to have no effect on test outcome
        CI - CI configuration files
    """
    parts = filename.split(os.sep)
    if parts[0] in ['.jenkins', '.circleci']:
        return 'CI'
    if parts[0] in ['docs', 'scripts', 'CODEOWNERS', 'README.md']:
        return 'NONE'
    elif parts[0] == 'torch':
        if parts[-1].endswith('.py') or parts[-1].endswith('.pyi'):
            return 'TORCH'
    elif parts[0] == 'caffe2':
        if parts[-1].endswith('.py') or parts[-1].endswith('.pyi'):
            return 'CAFFE2'
    elif parts[0] == 'test':
        if parts[-1].endswith('.py') or parts[-1].endswith('.pyi'):
            return 'TEST'

    return 'UNKNOWN'


def log_test_reason(file_type, filename, test, options):
    if options.verbose:
        print_to_stderr(
            'Determination found {} file {} -- running {}'.format(
                file_type,
                filename,
                test,
            )
        )


def get_dep_modules(test):
    # Cache results in case of repitition
    if test in _DEP_MODULES_CACHE:
        return _DEP_MODULES_CACHE[test]

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_location = os.path.join(repo_root, 'test', test + '.py')
    finder = modulefinder.ModuleFinder(
        # Ideally exclude all third party modules, to speed up calculation.
        excludes=[
            'scipy',
            'numpy',
            'numba',
            'multiprocessing',
            'sklearn',
            'setuptools',
            'hypothesis',
            'llvmlite',
            'joblib',
            'email',
            'importlib',
            'unittest',
            'urllib',
            'json',
            'collections',
        ],
    )
    # HACK: some platforms default to ascii, so we can't just run_script :(
    if PY2:
        finder.run_script(test_location)
    else:
        with open(test_location, 'r', encoding='utf-8') as fp:
            finder.load_module('__main__', fp, test_location, ('', 'r', 1))

    dep_modules = set(finder.modules.keys())
    _DEP_MODULES_CACHE[test] = dep_modules
    return dep_modules


def determine_target(test, touched_files, options):
    test = parse_test_module(test)
    # Some tests are faster to execute than to determine.
    if test not in SLOW_TESTS:
        if options.verbose:
            print_to_stderr('Running {} without determination'.format(test))
        return True
    # HACK: "no_ninja" is not a real module
    if test.endswith('_no_ninja'):
        test = test[:(-1 * len('_no_ninja'))]
    if test.endswith('_ninja'):
        test = test[:(-1 * len('_ninja'))]

    dep_modules = get_dep_modules(test)

    for touched_file in touched_files:
        file_type = test_impact_of_file(touched_file)
        if file_type == 'NONE':
            continue
        elif file_type == 'CI':
            # Force all tests to run if any change is made to the CI
            # configurations.
            log_test_reason(file_type, touched_file, test, options)
            return True
        elif file_type == 'UNKNOWN':
            # Assume uncategorized source files can affect every test.
            log_test_reason(file_type, touched_file, test, options)
            return True
        elif file_type in ['TORCH', 'CAFFE2', 'TEST']:
            parts = os.path.splitext(touched_file)[0].split(os.sep)
            touched_module = ".".join(parts)
            # test/ path does not have a "test." namespace
            if touched_module.startswith('test.'):
                touched_module = touched_module.split('test.')[1]
            if (
                touched_module in dep_modules
                or touched_module == test.replace('/', '.')
            ):
                log_test_reason(file_type, touched_file, test, options)
                return True

    # If nothing has determined the test has run, don't run the test.
    if options.verbose:
        print_to_stderr('Determination is skipping {}'.format(test))

    return False


def main():
    options = parse_args()
    executable = get_executable_command(options)  # this is a list
    print_to_stderr('Test executor: {}'.format(executable))
    test_directory = os.path.dirname(os.path.abspath(__file__))
    selected_tests = get_selected_tests(options)

    if options.verbose:
        print_to_stderr('Selected tests: {}'.format(', '.join(selected_tests)))

    if options.coverage:
        shell(['coverage', 'erase'])

    if options.jit:
        selected_tests = filter(lambda test_name: "jit" in test_name, TESTS)

    if options.determine_from is not None and os.path.exists(options.determine_from):
        with open(options.determine_from, 'r') as fh:
            touched_files = [
                os.path.normpath(name.strip()) for name in fh.read().split('\n')
                if len(name.strip()) > 0
            ]
        # HACK: Ensure the 'test' paths can be traversed by Modulefinder
        sys.path.append('test')
        selected_tests = [
            test for test in selected_tests
            if determine_target(test, touched_files, options)
        ]
        sys.path.remove('test')

    for test in selected_tests:

        test_module = parse_test_module(test)

        # Printing the date here can help diagnose which tests are slow
        print_to_stderr('Running {} ... [{}]'.format(test, datetime.now()))
        handler = CUSTOM_HANDLERS.get(test, run_test)
        return_code = handler(executable, test_module, test_directory, options)
        assert isinstance(return_code, int) and not isinstance(
            return_code, bool), 'Return code should be an integer'
        if return_code != 0:
            message = '{} failed!'.format(test)
            if return_code < 0:
                # subprocess.Popen returns the child process' exit signal as
                # return code -N, where N is the signal number.
                signal_name = SIGNALS_TO_NAMES_DICT[-return_code]
                message += ' Received signal: {}'.format(signal_name)
            raise RuntimeError(message)
    if options.coverage:
        shell(['coverage', 'combine'])
        shell(['coverage', 'html'])


if __name__ == '__main__':
    main()

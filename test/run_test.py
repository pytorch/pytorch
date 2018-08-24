#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile

import torch
from torch.utils import cpp_extension
from common import TEST_WITH_ROCM

TESTS = [
    'autograd',
    'cpp_extensions',
    'c10d',
    'cuda',
    'dataloader',
    'distributed',
    'distributions',
    'indexing',
    'jit',
    'legacy_nn',
    'multiprocessing',
    'nccl',
    'nn',
    'optim',
    'sparse',
    'torch',
    'utils',
]

WINDOWS_BLACKLIST = [
    'distributed',
]

ROCM_BLACKLIST = [
    'c10d',
    'cpp_extensions',
    'distributed',
    'distributions',
    'multiprocessing',
    'nccl',
    'utils',
]

DISTRIBUTED_TESTS_CONFIG = {
    'tcp': {
        'WORLD_SIZE': '3'
    },
    'gloo': {
        'WORLD_SIZE': '2' if torch.cuda.device_count() == 2 else '3'
    },
    'nccl': {
        'WORLD_SIZE': '2'
    },
    'mpi': {
        'WORLD_SIZE': '3'
    },
}

# https://stackoverflow.com/questions/2549939/get-signal-names-from-numbers-in-python
SIGNALS_TO_NAMES_DICT = dict((getattr(signal, n), n) for n in dir(signal)
                             if n.startswith('SIG') and '_' not in n)


def print_to_stderr(message):
    print(message, file=sys.stderr)


def shell(command, cwd):
    sys.stdout.flush()
    sys.stderr.flush()
    return subprocess.call(
        shlex.split(command), universal_newlines=True, cwd=cwd)


def get_shell_output(command):
    return subprocess.check_output(shlex.split(command)).decode().strip()


def run_test(python, test_module, test_directory, options):
    verbose = '--verbose' if options.verbose else ''
    # Can't call `python -m unittest test_*` here because it doesn't run code
    # in `if __name__ == '__main__': `. So call `python test_*.py` instead.
    return shell('{} {}.py {}'.format(python, test_module, verbose),
                 test_directory)


def test_cpp_extensions(python, test_module, test_directory, options):
    try:
        cpp_extension.verify_ninja_availability()
    except RuntimeError:
        print(
            'Ninja is not available. Skipping C++ extensions test. '
            "Install ninja with 'pip install ninja' or 'conda install ninja'.")
        return 0
    return_code = shell('{} setup.py install --root ./install'.format(python),
                        os.path.join(test_directory, 'cpp_extensions'))
    if return_code != 0:
        return return_code

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
        return run_test(python, test_module, test_directory, options)
    finally:
        os.environ['PYTHONPATH'] = python_path


def test_distributed(python, test_module, test_directory, options):
    mpi_available = subprocess.call('command -v mpiexec', shell=True) == 0
    if options.verbose and not mpi_available:
        print_to_stderr(
            'MPI not available -- MPI backend tests will be skipped')
    for backend, env_vars in DISTRIBUTED_TESTS_CONFIG.items():
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
                init_method = 'file://{}/shared_init_file'.format(tmp_dir)
                os.environ['INIT_METHOD'] = init_method
            try:
                os.mkdir(os.path.join(tmp_dir, 'barrier'))
                os.mkdir(os.path.join(tmp_dir, 'test_dir'))
                if backend == 'mpi':
                    # test mpiexec for --noprefix option
                    devnull = open(os.devnull, 'w')
                    noprefix_opt = '--noprefix' if subprocess.call(
                        'mpiexec -n 1 --noprefix bash -c ""', shell=True,
                        stdout=devnull, stderr=subprocess.STDOUT) == 0 else ''

                    mpiexec = 'mpiexec -n 3 {} {}'.format(noprefix_opt, python)

                    return_code = run_test(mpiexec, test_module,
                                           test_directory, options)
                else:
                    return_code = run_test(python, test_module, test_directory,
                                           options)
                if return_code != 0:
                    return return_code
            finally:
                shutil.rmtree(tmp_dir)
    return 0


CUSTOM_HANDLERS = {
    'cpp_extensions': test_cpp_extensions,
    'distributed': test_distributed,
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
        '-p', '--python', help='the python interpreter to execute tests with')
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
        '--ignore-win-blacklist',
        action='store_true',
        help='always run blacklisted windows tests')
    return parser.parse_args()


def get_python_command(options):
    if options.coverage:
        return 'coverage run --parallel-mode --source torch'
    elif options.python:
        return options.python
    else:
        return os.environ.get('PYCMD', 'python')


def find_test_index(test, selected_tests, find_last_index=False):
    """Find the index of the first or last occurrence of a given test/test module in the list of seleceted tests.

    This function is used to determine the indexes when slicing the list of selected tests when
    ``options.first``(:attr:`find_last_index`=False) and/or ``options.last``(:attr:`find_last_index`=True) are used.

    :attr:`selected_tests` can be a list that contains multiple consequent occurrences of tests
    as part of the same test module, e.g.:

    ```
    selected_tests = ['autograd', 'cuda', **'torch.TestTorch.test_acos',
                     'torch.TestTorch.test_tan', 'torch.TestTorch.test_add'**, 'utils']
    ```

    If :attr:`test`='torch' and :attr:`find_last_index`=False result should be **2**.
    If :attr:`test`='torch' and :attr:`find_last_index`=True result should be **4**.

    Arguments:
        test (str): Name of test to lookup
        selected_tests (list): List of tests
        find_last_index (bool, optional): should we lookup the index of first or last
            occurrence (first is default)

    Returns:
        index of the first or last occurance of the given test
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
    tests_copy = selected_tests[:]
    for exclude_test in exclude_list:
        for test in tests_copy:
            if test.startswith(exclude_test):
                if exclude_message is not None:
                    print_to_stderr('Excluding {} {}'.format(test, exclude_message))
                selected_tests.remove(test)
    return selected_tests


def get_selected_tests(options):
    selected_tests = options.include

    if options.first:
        first_index = find_test_index(options.first, selected_tests)
        selected_tests = selected_tests[first_index:]

    if options.last:
        last_index = find_test_index(options.last, selected_tests, find_last_index=True)
        selected_tests = selected_tests[:last_index + 1]

    selected_tests = exclude_tests(options.exclude, selected_tests)

    if sys.platform == 'win32' and not options.ignore_win_blacklist:
        ostype = os.environ.get('MSYSTEM')
        target_arch = os.environ.get('VSCMD_ARG_TGT_ARCH')
        if ostype != 'MINGW64' or target_arch != 'x64':
            WINDOWS_BLACKLIST.append('cpp_extensions')

        selected_tests = exclude_tests(WINDOWS_BLACKLIST, selected_tests, 'on Windows')

    elif TEST_WITH_ROCM:
        selected_tests = exclude_tests(ROCM_BLACKLIST, selected_tests, 'on ROCm')

    return selected_tests


def main():
    options = parse_args()
    python = get_python_command(options)
    test_directory = os.path.dirname(os.path.abspath(__file__))
    selected_tests = get_selected_tests(options)

    if options.verbose:
        print_to_stderr('Selected tests: {}'.format(', '.join(selected_tests)))

    if options.coverage:
        shell('coverage erase')

    for test in selected_tests:
        test_name = 'test_{}'.format(test)
        test_module = parse_test_module(test)

        print_to_stderr('Running {} ...'.format(test_name))
        handler = CUSTOM_HANDLERS.get(test_module, run_test)
        return_code = handler(python, test_name, test_directory, options)
        assert isinstance(return_code, int) and not isinstance(
            return_code, bool), 'Return code should be an integer'
        if return_code != 0:
            message = '{} failed!'.format(test_name)
            if return_code < 0:
                # subprocess.Popen returns the child process' exit signal as
                # return code -N, where N is the signal number.
                signal_name = SIGNALS_TO_NAMES_DICT[-return_code]
                message += ' Received signal: {}'.format(signal_name)
            raise RuntimeError(message)

    if options.coverage:
        shell('coverage combine')
        shell('coverage html')


if __name__ == '__main__':
    main()

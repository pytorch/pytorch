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

TESTS = [
    'autograd',
    'cpp_extensions',
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
    'mpi': {},
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
    return shell('{} {} {}'.format(python, test_module, verbose),
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
        if sys.platform == 'win32':
            install_directory = os.path.join(cpp_extensions, 'install')
            install_directories = get_shell_output(
                'where -r "{}" *.pyd'.format(install_directory)).split('\r\n')

            assert install_directories, 'install_directory must not be empty'

            if len(install_directories) >= 1:
                install_directory = install_directories[0]

            install_directory = os.path.dirname(install_directory)
            split_char = ';'
        else:
            install_directory = get_shell_output(
                "find {}/install -name *-packages".format(cpp_extensions))
            split_char = ':'

        assert install_directory, 'install_directory must not be empty'
        install_directory = os.path.join(test_directory, install_directory)
        os.environ['PYTHONPATH'] = '{}{}{}'.format(install_directory,
                                                   split_char, python_path)
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
                    mpiexec = 'mpiexec -n 3 --noprefix {}'.format(python)
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
        choices=TESTS,
        default=TESTS,
        metavar='TESTS',
        help='select a set of tests to include (defaults to ALL tests)')
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


def get_selected_tests(options):
    selected_tests = options.include
    for test in options.exclude:
        if test in selected_tests:
            selected_tests.remove(test)

    if options.first:
        first_index = selected_tests.index(options.first)
        selected_tests = selected_tests[first_index:]

    if options.last:
        last_index = selected_tests.index(options.last)
        selected_tests = selected_tests[:last_index + 1]

    if sys.platform == 'win32' and not options.ignore_win_blacklist:
        ostype = os.environ.get('MSYSTEM')
        target_arch = os.environ.get('VSCMD_ARG_TGT_ARCH')
        if ostype != 'MINGW64' or target_arch != 'x64':
            WINDOWS_BLACKLIST.append('cpp_extensions')

        for test in WINDOWS_BLACKLIST:
            if test in selected_tests:
                print_to_stderr('Excluding {} on Windows'.format(test))
                selected_tests.remove(test)

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
        test_module = 'test_{}.py'.format(test)
        print_to_stderr('Running {} ...'.format(test_module))
        handler = CUSTOM_HANDLERS.get(test, run_test)
        return_code = handler(python, test_module, test_directory, options)
        assert isinstance(return_code, int) and not isinstance(
            return_code, bool), 'Return code should be an integer'
        if return_code != 0:
            message = '{} failed!'.format(test_module)
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

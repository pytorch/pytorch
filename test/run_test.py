#!/usr/bin/env python

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import torch

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
    'cpp_extensions',
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


def shell(command, cwd):
    popen = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        universal_newlines=True,
        cwd=cwd,
        shell=True)
    for stdout_line in iter(popen.stdout.readline, ''):
        print(stdout_line.strip('\n'))
    popen.stdout.close()
    return_code = popen.wait()
    return return_code == 0


def get_shell_output(command):
    return subprocess.check_output(command, shell=True).decode().strip()


def run_test(python, test_module, test_directory, verbose):
    verbose = '--verbose' if verbose else ''
    return shell('{} {} {}'.format(python, test_module, verbose), test_directory)


def test_cpp_extensions(python, test_module, test_directory, verbose):
    if not shell('{} setup.py install --root ./install'.format(python),
          os.path.join(test_directory, 'cpp_extensions')):
        return False

    python_path = os.environ.get('PYTHONPATH', '')
    try:
        install_directory = get_shell_output(
            "find cpp_extensions/install -name '*-packages'")
        install_directory = os.path.join(test_directory, install_directory)
        os.environ['PYTHONPATH'] = '{}:{}'.format(install_directory, python_path)
        return run_test(python, test_module, test_directory, verbose)
    finally:
        os.environ['PYTHONPATH'] = python_path

def test_distributed(python, test_module, test_directory, verbose):
    mpi_available = subprocess.call('command -v mpiexec', shell=True) == 0
    if verbose and not mpi_available:
        print('MPI not available -- MPI backend tests will be skipped')
    for backend, env_vars in DISTRIBUTED_TESTS_CONFIG.items():
        if backend == 'mpi' and not mpi_available:
            continue
        for with_init in {True, False}:
            tmp_dir = tempfile.mkdtemp()
            with_init_message = ' with file init_method' if with_init else ''
            if verbose:
                print('Running distributed tests for the {} backend{}'.format(
                    backend, with_init_message))
            os.environ['TEMP_DIR'] = tmp_dir
            os.environ['BACKEND'] = backend
            os.environ['INIT_METHOD'] = 'env://'
            os.environ.update(env_vars)
            if with_init:
                init_method = 'file://{}/shared_init_file'.format(tmp_dir)
                os.environ['INIT_METHOD'] = init_method
            try:
                os.mkdir(os.path.join(tmp_dir, 'barrier'))
                os.mkdir(os.path.join(tmp_dir, 'test_dir'))
                if backend == 'mpi':
                    mpiexec = 'mpiexec -n 3 --noprefix {}'.format(python)
                    if not run_test(mpiexec, test_module, test_directory, verbose):
                        return False
                else:
                    if not run_test(python, test_module, test_directory, verbose):
                        return False
            finally:
                shutil.rmtree(tmp_dir)
    return True

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
        for test in WINDOWS_BLACKLIST:
            if test in selected_tests:
                print('Excluding {} on Windows'.format(test))
                selected_tests.remove(test)

    return selected_tests


def main():
    options = parse_args()
    python = get_python_command(options)
    test_directory = os.path.dirname(os.path.abspath(__file__))
    selected_tests = get_selected_tests(options)
    if options.verbose:
        print('Selected tests: {}'.format(', '.join(selected_tests)))

    if options.coverage:
        shell('coverage erase')

    for test in selected_tests:
        test_module = 'test_{}.py'.format(test)
        print('Running {} ...'.format(test_module))
        handler = CUSTOM_HANDLERS.get(test, run_test)
        if not handler(python, test_module, test_directory, options.verbose):
            break

    if options.coverage:
        shell('coverage combine')
        shell('coverage html')


if __name__ == '__main__':
    main()

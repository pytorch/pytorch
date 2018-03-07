#!/usr/bin/env python

import argparse
import os
import subprocess
import sys

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


def shell(command, cwd):
    popen = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        universal_newlines=True,
        cwd=cwd,
        shell=True)
    for stdout_line in iter(popen.stdout.readline, ''):
        print(stdout_line.strip())
    popen.stdout.close()
    return_code = popen.wait()
    return return_code == 0


def get_shell_output(command):
    return subprocess.check_output(command, shell=True).decode().strip()


def run_test(python, test_module, cwd, verbose):
    verbose = '--verbose' if verbose else ''
    shell('{} {} {}'.format(python, test_module, verbose), cwd)


def test_cpp_extensions(python, test_module, test_directory, verbose):
    shell('{} setup.py install --root ./install'.format(python),
          os.path.join(test_directory, 'cpp_extensions'))

    python_path = os.environ.get('PYTHONPATH', '')
    install_directory = get_shell_output(
        "find cpp_extensions/install -name '*-packages'")
    install_directory = os.path.join(test_directory, install_directory)
    os.environ['PYTHONPATH'] = '{}:{}'.format(install_directory, python_path)
    run_test(python, test_module, test_directory, verbose)
    os.environ['PYTHONPATH'] = python_path


def test_distributed(python, test_module, cwd, verbose):
    os.environ['PYCMD'] = python
    shell('./run_distributed_tests.sh', cwd)


CUSTOM_HANDLERS = {
    'cpp_extensions': test_cpp_extensions,
    'distributed': test_distributed,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-p', '--python')
    parser.add_argument('-c', '--coverage', action='store_true')
    parser.add_argument(
        '-i', '--include', nargs='+', choices=TESTS, default=TESTS)
    parser.add_argument(
        '-x', '--exclude', nargs='+', choices=TESTS, default=[])
    parser.add_argument('-f', '--first', choices=TESTS)
    parser.add_argument('-l', '--last', choices=TESTS)
    parser.add_argument('--with-windows', action='store_true')
    parser.add_argument('--test-dir')
    return parser.parse_args()


def get_python_command(options):
    if options.coverage:
        return 'coverage run --parallel-mode --source torch'
    elif options.python:
        return options.python
    else:
        return os.environ.get('PYCMD', 'python')


def get_selected_tests(options):
    selected_tests = list(sorted(set(options.include) - set(options.exclude)))

    if options.first:
        first_index = selected_tests.index(options.first)
        selected_tests = selected_tests[first_index:]

    if options.last:
        last_index = selected_tests.index(options.last)
        selected_tests = selected_tests[:last_index + 1]

    if sys.platform == 'win32' and not options.with_windows:
        for test in WINDOWS_BLACKLIST:
            if test in selected_tests:
                print('Excluding {} on Windows'.format(test))
                selected_tests.remove(test)

    return selected_tests


def get_test_directory(options):
    return options.test_dir or os.path.dirname(os.path.abspath(__file__))


def main():
    options = parse_args()
    python = get_python_command(options)
    selected_tests = get_selected_tests(options)
    print('Selected tests: {}'.format(', '.join(selected_tests)))
    test_directory = get_test_directory(options)

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

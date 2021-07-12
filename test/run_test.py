#!/usr/bin/env python3

import argparse
import copy
from datetime import datetime
import modulefinder
import os
import shutil
import signal
import subprocess
import sys
import tempfile

import torch
from torch.utils import cpp_extension
from torch.testing._internal.common_utils import FILE_SCHEMA, IS_IN_CI, TEST_WITH_ROCM, shell, set_cwd
import torch.distributed as dist
from typing import Dict, Optional, List

try:
    # using tools/ to optimize test run.
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from tools.testing.test_selections import (
        export_S3_test_times,
        get_shard_based_on_S3,
        get_slow_tests_based_on_S3,
        get_specified_test_cases,
        get_reordered_tests,
        get_test_case_configs,
    )
    HAVE_TEST_SELECTION_TOOLS = True
except ImportError:
    HAVE_TEST_SELECTION_TOOLS = False
    print("Unable to import test_selections from tools/testing. Running without test selection stats...")


TESTS = [
    'test_import_time',
    'test_public_bindings',
    'test_type_hints',
    'test_ao_sparsity',
    'test_autograd',
    'benchmark_utils/test_benchmark_utils',
    'test_binary_ufuncs',
    'test_bundled_inputs',
    'test_complex',
    'test_cpp_api_parity',
    'test_cpp_extensions_aot_no_ninja',
    'test_cpp_extensions_aot_ninja',
    'test_cpp_extensions_jit',
    'distributed/test_c10d_common',
    'distributed/test_c10d_gloo',
    'distributed/test_c10d_nccl',
    'distributed/test_jit_c10d',
    'distributed/test_c10d_spawn_gloo',
    'distributed/test_c10d_spawn_nccl',
    'distributed/test_store',
    'distributed/test_pg_wrapper',
    'test_cuda',
    'test_jit_cuda_fuser',
    'test_cuda_primary_ctx',
    'test_dataloader',
    'test_datapipe',
    'distributed/test_data_parallel',
    'distributed/test_distributed_fork',
    'distributed/test_distributed_spawn',
    'distributions/test_constraints',
    'distributions/test_distributions',
    'test_dispatch',
    'test_foreach',
    'test_indexing',
    'test_jit',
    'test_linalg',
    'test_logging',
    'test_mkldnn',
    'test_model_dump',
    'test_module_init',
    'test_multiprocessing',
    'test_multiprocessing_spawn',
    'distributed/test_nccl',
    'test_native_functions',
    'test_numba_integration',
    'test_nn',
    'test_ops',
    'test_optim',
    'test_pytree',
    'test_mobile_optimizer',
    'test_set_default_mobile_cpu_allocator',
    'test_xnnpack_integration',
    'test_vulkan',
    'test_sparse',
    'test_sparse_csr',
    'test_quantization',
    'test_pruning_op',
    'test_spectral_ops',
    'test_serialization',
    'test_shape_ops',
    'test_show_pickle',
    'test_sort_and_select',
    'test_tensor_creation_ops',
    'test_testing',
    'test_torch',
    'test_type_info',
    'test_unary_ufuncs',
    'test_utils',
    'test_view_ops',
    'test_vmap',
    'test_namedtuple_return_api',
    'test_numpy_interop',
    'test_jit_profiling',
    'test_jit_legacy',
    'test_jit_fuser_legacy',
    'test_tensorboard',
    'test_namedtensor',
    'test_reductions',
    'test_type_promotion',
    'test_jit_disabled',
    'test_function_schema',
    'test_overrides',
    'test_jit_fuser_te',
    'test_tensorexpr',
    'test_tensorexpr_pybind',
    'test_openmp',
    'test_profiler',
    "distributed/test_launcher",
    'distributed/nn/jit/test_instantiator',
    'distributed/rpc/test_faulty_agent',
    'distributed/rpc/test_tensorpipe_agent',
    'distributed/rpc/cuda/test_tensorpipe_agent',
    'test_determination',
    'test_futures',
    'test_fx',
    'test_fx_experimental',
    'test_functional_autograd_benchmark',
    'test_package',
    'test_license',
    'distributed/pipeline/sync/skip/test_api',
    'distributed/pipeline/sync/skip/test_gpipe',
    'distributed/pipeline/sync/skip/test_inspect_skip_layout',
    'distributed/pipeline/sync/skip/test_leak',
    'distributed/pipeline/sync/skip/test_portal',
    'distributed/pipeline/sync/skip/test_stash_pop',
    'distributed/pipeline/sync/skip/test_tracker',
    'distributed/pipeline/sync/skip/test_verify_skippables',
    'distributed/pipeline/sync/test_balance',
    'distributed/pipeline/sync/test_bugs',
    'distributed/pipeline/sync/test_checkpoint',
    'distributed/pipeline/sync/test_copy',
    'distributed/pipeline/sync/test_deferred_batch_norm',
    'distributed/pipeline/sync/test_dependency',
    'distributed/pipeline/sync/test_inplace',
    'distributed/pipeline/sync/test_microbatch',
    'distributed/pipeline/sync/test_phony',
    'distributed/pipeline/sync/test_pipe',
    'distributed/pipeline/sync/test_pipeline',
    'distributed/pipeline/sync/test_stream',
    'distributed/pipeline/sync/test_transparency',
    'distributed/pipeline/sync/test_worker',
    'distributed/optim/test_zero_redundancy_optimizer',
    'distributed/elastic/timer/api_test',
    'distributed/elastic/timer/local_timer_example',
    'distributed/elastic/timer/local_timer_test',
    'distributed/elastic/events/lib_test',
    'distributed/elastic/metrics/api_test',
    'distributed/elastic/utils/logging_test',
    'distributed/elastic/utils/util_test',
    'distributed/elastic/utils/distributed_test',
    'distributed/elastic/multiprocessing/api_test',
    'distributed/_sharding_spec/test_sharding_spec',
]

# Tests need to be run with pytest.
USE_PYTEST_LIST = [
    'distributed/pipeline/sync/skip/test_api',
    'distributed/pipeline/sync/skip/test_gpipe',
    'distributed/pipeline/sync/skip/test_inspect_skip_layout',
    'distributed/pipeline/sync/skip/test_leak',
    'distributed/pipeline/sync/skip/test_portal',
    'distributed/pipeline/sync/skip/test_stash_pop',
    'distributed/pipeline/sync/skip/test_tracker',
    'distributed/pipeline/sync/skip/test_verify_skippables',
    'distributed/pipeline/sync/test_balance',
    'distributed/pipeline/sync/test_bugs',
    'distributed/pipeline/sync/test_checkpoint',
    'distributed/pipeline/sync/test_copy',
    'distributed/pipeline/sync/test_deferred_batch_norm',
    'distributed/pipeline/sync/test_dependency',
    'distributed/pipeline/sync/test_inplace',
    'distributed/pipeline/sync/test_microbatch',
    'distributed/pipeline/sync/test_phony',
    'distributed/pipeline/sync/test_pipe',
    'distributed/pipeline/sync/test_pipeline',
    'distributed/pipeline/sync/test_stream',
    'distributed/pipeline/sync/test_transparency',
    'distributed/pipeline/sync/test_worker',
    'distributions/test_constraints',
    'distributions/test_transforms',
    'distributions/test_utils',
    'test_typing',
    "distributed/elastic/events/lib_test",
    "distributed/elastic/agent/server/test/api_test",
]

WINDOWS_BLOCKLIST = [
    'distributed/nn/jit/test_instantiator',
    'distributed/rpc/test_faulty_agent',
    'distributed/rpc/test_tensorpipe_agent',
    'distributed/rpc/cuda/test_tensorpipe_agent',
    'distributed/test_distributed_fork',
    'distributed/pipeline/sync/skip/test_api',
    'distributed/pipeline/sync/skip/test_gpipe',
    'distributed/pipeline/sync/skip/test_inspect_skip_layout',
    'distributed/pipeline/sync/skip/test_leak',
    'distributed/pipeline/sync/skip/test_portal',
    'distributed/pipeline/sync/skip/test_stash_pop',
    'distributed/pipeline/sync/skip/test_tracker',
    'distributed/pipeline/sync/skip/test_verify_skippables',
    'distributed/pipeline/sync/test_balance',
    'distributed/pipeline/sync/test_bugs',
    'distributed/pipeline/sync/test_checkpoint',
    'distributed/pipeline/sync/test_copy',
    'distributed/pipeline/sync/test_deferred_batch_norm',
    'distributed/pipeline/sync/test_dependency',
    'distributed/pipeline/sync/test_inplace',
    'distributed/pipeline/sync/test_microbatch',
    'distributed/pipeline/sync/test_phony',
    'distributed/pipeline/sync/test_pipe',
    'distributed/pipeline/sync/test_pipeline',
    'distributed/pipeline/sync/test_stream',
    'distributed/pipeline/sync/test_transparency',
    'distributed/pipeline/sync/test_worker',
    'distributed/optim/test_zero_redundancy_optimizer',
    "distributed/elastic/agent/server/test/api_test",
    'distributed/elastic/multiprocessing/api_test',
]

ROCM_BLOCKLIST = [
    'distributed/nn/jit/test_instantiator',
    'distributed/rpc/test_faulty_agent',
    'distributed/rpc/test_tensorpipe_agent',
    'distributed/rpc/cuda/test_tensorpipe_agent',
    'test_determination',
    'test_multiprocessing',
    'test_jit_legacy',
    'test_type_hints',
    'test_openmp',
]

RUN_PARALLEL_BLOCKLIST = [
    'test_cpp_extensions_jit',
    'test_jit_disabled',
    'test_mobile_optimizer',
    'test_multiprocessing',
    'test_multiprocessing_spawn',
    'test_namedtuple_return_api',
    'test_overrides',
    'test_show_pickle',
    'test_tensorexpr',
    'test_cuda_primary_ctx',
] + [test for test in TESTS if test.startswith('distributed/')]

WINDOWS_COVERAGE_BLOCKLIST = [
]


# These tests are slow enough that it's worth calculating whether the patch
# touched any related files first. This list was manually generated, but for every
# run with --determine-from, we use another generated list based on this one and the
# previous test stats.
TARGET_DET_LIST = [
    'distributions/test_distributions',
    'test_nn',
    'test_autograd',
    'test_cpp_extensions_jit',
    'test_jit_legacy',
    'test_dataloader',
    'test_overrides',
    'test_linalg',
    'test_jit',
    'test_jit_profiling',
    'test_torch',
    'test_binary_ufuncs',
    'test_numpy_interop',
    'test_reductions',
    'test_shape_ops',
    'test_sort_and_select',
    'test_testing',
    'test_view_ops',
    'distributed/nn/jit/test_instantiator',
    'distributed/test_distributed_fork',
    'distributed/rpc/test_tensorpipe_agent',
    'distributed/rpc/cuda/test_tensorpipe_agent',
    'distributed/algorithms/ddp_comm_hooks/test_ddp_hooks',
    'distributed/test_distributed_spawn',
    'test_cuda',
    'test_cuda_primary_ctx',
    'test_cpp_extensions_aot_ninja',
    'test_cpp_extensions_aot_no_ninja',
    'test_serialization',
    'test_optim',
    'test_utils',
    'test_multiprocessing',
    'test_tensorboard',
    'distributed/test_c10d_common',
    'distributed/test_c10d_gloo',
    'distributed/test_c10d_nccl',
    'distributed/test_jit_c10d',
    'distributed/test_c10d_spawn_gloo',
    'distributed/test_c10d_spawn_nccl',
    'distributed/test_store',
    'distributed/test_pg_wrapper',
    'test_quantization',
    'test_pruning_op',
    'test_determination',
    'test_futures',
    'distributed/pipeline/sync/skip/test_api',
    'distributed/pipeline/sync/skip/test_gpipe',
    'distributed/pipeline/sync/skip/test_inspect_skip_layout',
    'distributed/pipeline/sync/skip/test_leak',
    'distributed/pipeline/sync/skip/test_portal',
    'distributed/pipeline/sync/skip/test_stash_pop',
    'distributed/pipeline/sync/skip/test_tracker',
    'distributed/pipeline/sync/skip/test_verify_skippables',
    'distributed/pipeline/sync/test_balance',
    'distributed/pipeline/sync/test_bugs',
    'distributed/pipeline/sync/test_checkpoint',
    'distributed/pipeline/sync/test_copy',
    'distributed/pipeline/sync/test_deferred_batch_norm',
    'distributed/pipeline/sync/test_dependency',
    'distributed/pipeline/sync/test_inplace',
    'distributed/pipeline/sync/test_microbatch',
    'distributed/pipeline/sync/test_phony',
    'distributed/pipeline/sync/test_pipe',
    'distributed/pipeline/sync/test_pipeline',
    'distributed/pipeline/sync/test_stream',
    'distributed/pipeline/sync/test_transparency',
    'distributed/pipeline/sync/test_worker',
]

# the JSON file to store the S3 test stats
TEST_TIMES_FILE = '.pytorch-test-times.json'

# if a test file takes longer than 5 min, we add it to TARGET_DET_LIST
SLOW_TEST_THRESHOLD = 300

_DEP_MODULES_CACHE: Dict[str, set] = {}

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
    if dist.is_gloo_available():
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

PYTORCH_COLLECT_COVERAGE = bool(os.environ.get("PYTORCH_COLLECT_COVERAGE"))

ENABLE_PR_HISTORY_REORDERING = bool(os.environ.get("ENABLE_PR_HISTORY_REORDERING", "0") == "1")

JIT_EXECUTOR_TESTS = [
    'test_jit_cuda_fuser',
    'test_jit_profiling',
    'test_jit_legacy',
    'test_jit_fuser_legacy',
]

# Dictionary matching test modules (in TESTS) to lists of test cases (within that test_module) that would be run when
# options.run_specified_test_cases is enabled.
# For example:
# {
#   "test_nn": ["test_doubletensor_avg_pool3d", "test_share_memory", "test_hook_requires_grad"],
#   ...
# }
# then for test_nn.py, we would ONLY run test_doubletensor_avg_pool3d, test_share_memory, and test_hook_requires_grad.
SPECIFIED_TEST_CASES_DICT: Dict[str, List[str]] = {}

# The file from which the SPECIFIED_TEST_CASES_DICT will be filled, a CSV of test cases that would be run when
# options.run_specified_test_cases is enabled.
SPECIFIED_TEST_CASES_FILE: str = '.pytorch_specified_test_cases.csv'


def print_to_stderr(message):
    print(message, file=sys.stderr)


def get_test_case_args(test_module, using_pytest) -> List[str]:
    args = []
    # if test_module not specified or specified with '__all__' then run all tests
    if test_module not in SPECIFIED_TEST_CASES_DICT or '__all__' in SPECIFIED_TEST_CASES_DICT[test_module]:
        return args

    if using_pytest:
        args.append('-k')
        args.append(' or '.join(SPECIFIED_TEST_CASES_DICT[test_module]))
    else:
        for test in SPECIFIED_TEST_CASES_DICT[test_module]:
            args.append('-k')
            args.append(test)

    return args


def get_executable_command(options, allow_pytest, disable_coverage=False):
    if options.coverage and not disable_coverage:
        executable = ['coverage', 'run', '--parallel-mode', '--source=torch']
    else:
        executable = [sys.executable]
    if options.pytest:
        if allow_pytest:
            executable += ['-m', 'pytest']
        else:
            print_to_stderr('Pytest cannot be used for this test. Falling back to unittest.')
    return executable


def run_test(test_module, test_directory, options, launcher_cmd=None, extra_unittest_args=None):
    unittest_args = options.additional_unittest_args.copy()
    if options.verbose:
        unittest_args.append(f'-{"v"*options.verbose}')  # in case of pytest
    if test_module in RUN_PARALLEL_BLOCKLIST:
        unittest_args = [arg for arg in unittest_args if not arg.startswith('--run-parallel')]
    if extra_unittest_args:
        assert isinstance(extra_unittest_args, list)
        unittest_args.extend(extra_unittest_args)

    # If using pytest, replace -f with equivalent -x
    if options.pytest:
        unittest_args = [arg if arg != '-f' else '-x' for arg in unittest_args]
    elif IS_IN_CI:
        # use the downloaded test cases configuration, not supported in pytest
        unittest_args.extend(['--import-slow-tests', '--import-disabled-tests'])

    # Multiprocessing related tests cannot run with coverage.
    # Tracking issue: https://github.com/pytorch/pytorch/issues/50661
    disable_coverage = sys.platform == 'win32' and test_module in WINDOWS_COVERAGE_BLOCKLIST

    # Extra arguments are not supported with pytest
    executable = get_executable_command(options, allow_pytest=not extra_unittest_args,
                                        disable_coverage=disable_coverage)

    # TODO: move this logic into common_utils.py instead of passing in "-k" individually
    # The following logic for running specified tests will only run for non-distributed tests, as those are dispatched
    # to test_distributed and not run_test (this function)
    if options.run_specified_test_cases:
        unittest_args.extend(get_test_case_args(test_module, 'pytest' in executable))

    # Can't call `python -m unittest test_*` here because it doesn't run code
    # in `if __name__ == '__main__': `. So call `python test_*.py` instead.
    argv = [test_module + '.py'] + unittest_args

    command = (launcher_cmd or []) + executable + argv
    print_to_stderr('Executing {} ... [{}]'.format(command, datetime.now()))
    return shell(command, test_directory)


def test_cuda_primary_ctx(test_module, test_directory, options):
    return run_test(test_module, test_directory, options, extra_unittest_args=['--subprocess'])


def _test_cpp_extensions_aot(test_directory, options, use_ninja):
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
    from shutil import copyfile
    test_module = 'test_cpp_extensions_aot' + ('_ninja' if use_ninja else '_no_ninja')
    copyfile(test_directory + '/test_cpp_extensions_aot.py', test_directory + '/' + test_module + '.py')
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
        return run_test(test_module, test_directory, options)
    finally:
        os.environ['PYTHONPATH'] = python_path
        if os.path.exists(test_directory + '/' + test_module + '.py'):
            os.remove(test_directory + '/' + test_module + '.py')


def test_cpp_extensions_aot_ninja(test_module, test_directory, options):
    return _test_cpp_extensions_aot(test_directory, options, use_ninja=True)


def test_cpp_extensions_aot_no_ninja(test_module, test_directory, options):
    return _test_cpp_extensions_aot(test_directory, options, use_ninja=False)


def test_distributed(test_module, test_directory, options):
    # MPI tests are broken with Python-3.9
    mpi_available = subprocess.call('command -v mpiexec', shell=True) == 0 and sys.version_info < (3, 9)
    if options.verbose and not mpi_available:
        print_to_stderr(
            'MPI not available -- MPI backend tests will be skipped')
    config = DISTRIBUTED_TESTS_CONFIG
    for backend, env_vars in config.items():
        if sys.platform == 'win32' and backend != 'gloo':
            continue
        if backend == 'mpi' and not mpi_available:
            continue
        for with_init_file in {True, False}:
            if sys.platform == 'win32' and not with_init_file:
                continue
            tmp_dir = tempfile.mkdtemp()
            if options.verbose:
                init_str = "with {} init_method"
                with_init = init_str.format("file" if with_init_file else "env")
                print_to_stderr(
                    'Running distributed tests for the {} backend {}'.format(
                        backend, with_init))
            os.environ['TEMP_DIR'] = tmp_dir
            os.environ['BACKEND'] = backend
            os.environ['INIT_METHOD'] = 'env://'
            os.environ.update(env_vars)
            if with_init_file:
                if test_module in ["test_distributed_fork", "test_distributed_spawn"]:
                    init_method = f'{FILE_SCHEMA}{tmp_dir}/'
                else:
                    init_method = f'{FILE_SCHEMA}{tmp_dir}/shared_init_file'
                os.environ['INIT_METHOD'] = init_method
            try:
                os.mkdir(os.path.join(tmp_dir, 'barrier'))
                os.mkdir(os.path.join(tmp_dir, 'test_dir'))
                if backend == 'mpi':
                    # test mpiexec for --noprefix option
                    with open(os.devnull, 'w') as devnull:
                        allowrunasroot_opt = '--allow-run-as-root' if subprocess.call(
                            'mpiexec --allow-run-as-root -n 1 bash -c ""', shell=True,
                            stdout=devnull, stderr=subprocess.STDOUT) == 0 else ''
                        noprefix_opt = '--noprefix' if subprocess.call(
                            f'mpiexec {allowrunasroot_opt} -n 1 --noprefix bash -c ""', shell=True,
                            stdout=devnull, stderr=subprocess.STDOUT) == 0 else ''

                    mpiexec = ['mpiexec', '-n', '3', noprefix_opt, allowrunasroot_opt]

                    return_code = run_test(test_module, test_directory, options,
                                           launcher_cmd=mpiexec)
                else:
                    return_code = run_test(test_module, test_directory, options)
                if return_code != 0:
                    return return_code
            finally:
                shutil.rmtree(tmp_dir)
    return 0


CUSTOM_HANDLERS = {
    'test_cuda_primary_ctx': test_cuda_primary_ctx,
    'test_cpp_extensions_aot_no_ninja': test_cpp_extensions_aot_no_ninja,
    'test_cpp_extensions_aot_ninja': test_cpp_extensions_aot_ninja,
    'distributed/test_distributed_fork': test_distributed,
    'distributed/test_distributed_spawn': test_distributed,
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
        epilog='where TESTS is any of: {}'.format(', '.join(TESTS)),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-v',
        '--verbose',
        action='count',
        default=0,
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
        '-c', '--coverage', action='store_true', help='enable coverage',
        default=PYTORCH_COLLECT_COVERAGE)
    parser.add_argument(
        '-i',
        '--include',
        nargs='+',
        choices=TestChoices(TESTS),
        default=TESTS,
        metavar='TESTS',
        help='select a set of tests to include (defaults to ALL tests).'
             ' tests must be a part of the TESTS list defined in run_test.py')
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
        '--ignore-win-blocklist',
        action='store_true',
        help='always run blocklisted windows tests')
    parser.add_argument(
        '--determine-from',
        help='File of affected source filenames to determine which tests to run.')
    parser.add_argument(
        '--continue-through-error',
        action='store_true',
        help='Runs the full test suite despite one of the tests failing')
    parser.add_argument(
        'additional_unittest_args',
        nargs='*',
        help='additional arguments passed through to unittest, e.g., '
             'python run_test.py -i sparse -- TestSparse.test_factory_size_check')
    parser.add_argument(
        '--export-past-test-times',
        nargs='?',
        type=str,
        const=TEST_TIMES_FILE,
        help='dumps test times from previous S3 stats into a file, format JSON',
    )
    parser.add_argument(
        '--shard',
        nargs=2,
        type=int,
        help='runs a shard of the tests (taking into account other selections), e.g., '
        '--shard 2 3 will break up the selected tests into 3 shards and run the tests '
        'in the 2nd shard (the first number should not exceed the second)',
    )
    parser.add_argument(
        '--exclude-jit-executor',
        action='store_true',
        help='exclude tests that are run for a specific jit config'
    )
    parser.add_argument(
        '--run-specified-test-cases',
        nargs='?',
        type=str,
        const=SPECIFIED_TEST_CASES_FILE,
        help='load specified test cases file dumped from previous OSS CI stats, format CSV. '
        ' If all test cases should run for a <test_module> please add a single row: \n'
        ' test_filename,test_case_name\n'
        ' ...\n'
        ' <test_module>,__all__\n'
        ' ...\n'
        'how we use the stats will be based on option "--use-specified-test-cases-by".'
    )
    parser.add_argument(
        '--use-specified-test-cases-by',
        type=str,
        choices=['include', 'bring-to-front'],
        default='include',
        help='used together with option "--run-specified-test-cases". When specified test case '
        'file is set, this option allows the user to control whether to only run the specified test '
        'modules or to simply bring the specified modules to front and also run the remaining '
        'modules. Note: regardless of this option, we will only run the specified test cases '
        ' within a specified test module. For unspecified test modules with the bring-to-front '
        'option, all test cases will be run, as one may expect.',
    )
    return parser.parse_args()


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

    Args:
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
    if options.run_specified_test_cases:
        if options.use_specified_test_cases_by == 'include':
            options.include = list(SPECIFIED_TEST_CASES_DICT.keys())
        elif options.use_specified_test_cases_by == 'bring-to-front':
            options.bring_to_front = list(SPECIFIED_TEST_CASES_DICT.keys())

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

    if options.exclude_jit_executor:
        options.exclude.extend(JIT_EXECUTOR_TESTS)

    selected_tests = exclude_tests(options.exclude, selected_tests)

    if sys.platform == 'win32' and not options.ignore_win_blocklist:
        target_arch = os.environ.get('VSCMD_ARG_TGT_ARCH')
        if target_arch != 'x64':
            WINDOWS_BLOCKLIST.append('cpp_extensions_aot_no_ninja')
            WINDOWS_BLOCKLIST.append('cpp_extensions_aot_ninja')
            WINDOWS_BLOCKLIST.append('cpp_extensions_jit')
            WINDOWS_BLOCKLIST.append('jit')
            WINDOWS_BLOCKLIST.append('jit_fuser')

        selected_tests = exclude_tests(WINDOWS_BLOCKLIST, selected_tests, 'on Windows')

    elif TEST_WITH_ROCM:
        selected_tests = exclude_tests(ROCM_BLOCKLIST, selected_tests, 'on ROCm')

    if options.shard:
        assert len(options.shard) == 2, "Unexpected shard format"
        assert min(options.shard) > 0, "Shards must be positive numbers"
        which_shard, num_shards = options.shard
        assert which_shard <= num_shards, "Selected shard must be less than or equal to total number of shards"
        assert num_shards <= len(selected_tests), f"Number of shards must be less than {len(selected_tests)}"
        # TODO: fix this to use test_times_filename, but currently this is not working
        # because setting the export arg immeidately halts the test execution.
        selected_tests = get_shard_based_on_S3(which_shard, num_shards, selected_tests, TEST_TIMES_FILE)

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
    # Cache results in case of repetition
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
            # Modules below are excluded because they are hitting https://bugs.python.org/issue40350
            # Trigger AttributeError: 'NoneType' object has no attribute 'is_package'
            'mpl_toolkits',
            'google',
            'onnx',
            # Triggers RecursionError
            'mypy'
        ],
    )
    # HACK: some platforms default to ascii, so we can't just run_script :(
    with open(test_location, 'r', encoding='utf-8') as fp:
        finder.load_module('__main__', fp, test_location, ('', 'r', 1))

    dep_modules = set(finder.modules.keys())
    _DEP_MODULES_CACHE[test] = dep_modules
    return dep_modules


def determine_target(target_det_list, test, touched_files, options):
    test = parse_test_module(test)
    # Some tests are faster to execute than to determine.
    if test not in target_det_list:
        if options.verbose:
            print_to_stderr(f'Running {test} without determination')
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
        print_to_stderr(f'Determination is skipping {test}')

    return False


def run_test_module(test: str, test_directory: str, options) -> Optional[str]:
    test_module = parse_test_module(test)

    # Printing the date here can help diagnose which tests are slow
    print_to_stderr('Running {} ... [{}]'.format(test, datetime.now()))
    handler = CUSTOM_HANDLERS.get(test_module, run_test)
    return_code = handler(test_module, test_directory, options)
    assert isinstance(return_code, int) and not isinstance(
        return_code, bool), 'Return code should be an integer'
    if return_code == 0:
        return None

    message = f'{test} failed!'
    if return_code < 0:
        # subprocess.Popen returns the child process' exit signal as
        # return code -N, where N is the signal number.
        signal_name = SIGNALS_TO_NAMES_DICT[-return_code]
        message += f' Received signal: {signal_name}'
    return message


def main():
    options = parse_args()

    # TODO: move this export & download function in tools/ folder
    test_times_filename = options.export_past_test_times
    if test_times_filename:
        print(f'Exporting past test times from S3 to {test_times_filename}, no tests will be run.')
        export_S3_test_times(test_times_filename)
        return

    specified_test_cases_filename = options.run_specified_test_cases
    if specified_test_cases_filename:
        print(f'Loading specified test cases to run from {specified_test_cases_filename}.')
        global SPECIFIED_TEST_CASES_DICT
        SPECIFIED_TEST_CASES_DICT = get_specified_test_cases(specified_test_cases_filename, TESTS)

    test_directory = os.path.dirname(os.path.abspath(__file__))
    selected_tests = get_selected_tests(options)

    if options.verbose:
        print_to_stderr('Selected tests: {}'.format(', '.join(selected_tests)))

    if options.coverage and not PYTORCH_COLLECT_COVERAGE:
        shell(['coverage', 'erase'])

    if options.jit:
        selected_tests = filter(lambda test_name: "jit" in test_name, TESTS)

    if options.determine_from is not None and os.path.exists(options.determine_from):
        slow_tests = get_slow_tests_based_on_S3(TESTS, TARGET_DET_LIST, SLOW_TEST_THRESHOLD)
        print('Added the following tests to target_det tests as calculated based on S3:')
        print(slow_tests)
        with open(options.determine_from, 'r') as fh:
            touched_files = [
                os.path.normpath(name.strip()) for name in fh.read().split('\n')
                if len(name.strip()) > 0
            ]
        # HACK: Ensure the 'test' paths can be traversed by Modulefinder
        sys.path.append('test')
        selected_tests = [
            test for test in selected_tests
            if determine_target(TARGET_DET_LIST + slow_tests, test, touched_files, options)
        ]
        sys.path.remove('test')

    if IS_IN_CI:
        selected_tests = get_reordered_tests(selected_tests, ENABLE_PR_HISTORY_REORDERING)
        # downloading test cases configuration to local environment
        get_test_case_configs(dirpath=os.path.dirname(os.path.abspath(__file__)))

    has_failed = False
    failure_messages = []
    try:
        for test in selected_tests:
            options_clone = copy.deepcopy(options)
            if test in USE_PYTEST_LIST:
                options_clone.pytest = True
            err_message = run_test_module(test, test_directory, options_clone)
            if err_message is None:
                continue
            has_failed = True
            failure_messages.append(err_message)
            if not options_clone.continue_through_error:
                raise RuntimeError(err_message)
            print_to_stderr(err_message)
    finally:
        if options.coverage:
            from coverage import Coverage
            test_dir = os.path.dirname(os.path.abspath(__file__))
            with set_cwd(test_dir):
                cov = Coverage()
                if PYTORCH_COLLECT_COVERAGE:
                    cov.load()
                cov.combine(strict=False)
                cov.save()
                if not PYTORCH_COLLECT_COVERAGE:
                    cov.html_report()

    if options.continue_through_error and has_failed:
        for err in failure_messages:
            print_to_stderr(err)
        sys.exit(1)

if __name__ == '__main__':
    main()

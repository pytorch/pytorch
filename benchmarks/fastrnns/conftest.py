import os
import pytest  # noqa: F401
import torch


default_rnns = ['cudnn', 'aten', 'jit', 'jit_premul', 'jit_premul_bias', 'jit_simple',
                         'jit_multilayer', 'py']
default_cnns = ['resnet18', 'resnet18_jit', 'resnet50', 'resnet50_jit']
all_nets = default_rnns + default_cnns

def pytest_generate_tests(metafunc):
    # This creates lists of tests to generate, can be customized
    if metafunc.cls.__name__ == "TestBenchNetwork":
        metafunc.parametrize('net_name', all_nets, scope="class")
        metafunc.parametrize("executor", [metafunc.config.getoption("executor")], scope="class")
        metafunc.parametrize("fuser", [metafunc.config.getoption("fuser")], scope="class")


def pytest_addoption(parser):
    parser.addoption("--fuser", default="old", help="fuser to use for benchmarks")
    parser.addoption("--executor", default="legacy", help="executor to use for benchmarks")


def pytest_benchmark_update_machine_info(config, machine_info):
    machine_info['pytorch_version'] = torch.__version__
    machine_info['circle_build_num'] = os.environ.get("CIRCLE_BUILD_NUM")
    machine_info['circle_project_name'] = os.environ.get("CIRCLE_PROJECT_REPONAME")

import logging
from typing import Any

from cli.lib.common.git_helper import clone_external_repo
from cli.lib.common.pip_helper import pip_install_packages
from cli.lib.common.utils import run_command, temp_environ, working_directory


logger = logging.getLogger(__name__)


def sample_vllm_test_library():
    """
    Simple sample to unblock the vllm ci development, which is mimic to
    https://github.com/vllm-project/vllm/blob/main/.buildkite/test-pipeline.yaml
    see run_test_plan for more details
    """
    # TODO(elainewy): Read from yaml file to handle the env and tests for vllm
    return {
        "vllm_basic_correctness_test": {
            "title": "Basic Correctness Test",
            "id": "vllm_basic_correctness_test",
            "env_var": {
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            },
            "steps": [
                {
                    "command": "pytest -v -s basic_correctness/test_cumem.py",
                },
                {
                    "command": "pytest -v -s basic_correctness/test_basic_correctness.py",
                },
                {
                    "command": "pytest -v -s basic_correctness/test_cpu_offload.py",
                },
                {
                    "command": "pytest -v -s basic_correctness/test_preemption.py",
                    "env_var": {
                        "VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT": "1",
                    },
                },
            ],
        },
        "vllm_basic_models_test": {
            "title": "Basic models test",
            "id": "vllm_basic_models_test",
            "steps": [
                {"command": "pytest -v -s models/test_transformers.py"},
                {"command": "pytest -v -s models/test_registry.py"},
                {"command": "pytest -v -s models/test_utils.py"},
                {"command": "pytest -v -s models/test_vision.py"},
                {"command": "pytest -v -s models/test_initialization.py"},
            ],
        },
        "vllm_entrypoints_test": {
            "title": "Entrypoints Test ",
            "id": "vllm_entrypoints_test",
            "env_var": {
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            },
            "steps": [
                {
                    "command": " ".join(
                        [
                            "pytest",
                            "-v",
                            "-s",
                            "entrypoints/llm",
                            "--ignore=entrypoints/llm/test_lazy_outlines.py",
                            "--ignore=entrypoints/llm/test_generate.py",
                            "--ignore=entrypoints/llm/test_generate_multiple_loras.py",
                            "--ignore=entrypoints/llm/test_collective_rpc.py",
                        ]
                    )
                },
                {"command": "pytest -v -s entrypoints/llm/test_lazy_outlines.py"},
                {"command": "pytest -v -s entrypoints/llm/test_generate.py "},
                {
                    "command": "pytest -v -s entrypoints/llm/test_generate_multiple_loras.py"
                },
                {
                    "env_var": {"VLLM_USE_V1": "0"},
                    "command": "pytest -v -s entrypoints/offline_mode",
                },
            ],
        },
        "vllm_regression_test": {
            "title": "Regression Test",
            "id": "vllm_regression_test",
            "package_install": ["modelscope"],
            "steps": [
                {"command": "pytest -v -s test_regression.py"},
            ],
        },
    }


def run_test_plan(test_plan: str, test_target: str, tests_map: dict[str, Any]):
    """
    a method to run list of tests based on the test plan.
    """
    logger.info("run %s tests.....", test_target)
    if test_plan not in tests_map:
        raise RuntimeError(
            f"test {test_plan} not found, please add it to test plan pool"
        )
    tests = tests_map[test_plan]
    logger.info("Running tests: %s", tests["title"])

    pkgs = tests.get("package_install", [])
    if pkgs:
        logger.info("Installing packages: %s", pkgs)
        pip_install_packages(packages=pkgs, prefer_uv=True)
    with (
        temp_environ(tests.get("env_var", {})),
        working_directory(tests.get("working_directory", "tests")),
    ):
        failures = []
        for step in tests["steps"]:
            with temp_environ(step.get("env_var", {})):
                code = run_command(cmd=step["command"], check=False)
                if code != 0:
                    failures.append(step)
        if failures:
            logger.error("Failed tests: %s", failures)
            raise RuntimeError(f"{len(failures)} pytest runs failed: {failures}")
        logger.info("Done. All tests passed")


def clone_vllm(dst: str = "vllm"):
    clone_external_repo(
        target="vllm",
        repo="https://github.com/vllm-project/vllm.git",
        dst=dst,
        update_submodules=True,
    )

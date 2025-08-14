import logging
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cli.lib.common.git_helper import clone_external_repo
from cli.lib.common.pip_helper import pip_install_packages, pkg_exists
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
        # test plan:
        # required id, title, and steps
        # optional: env_var, package_install, working_directory
        # by default the working_drectory is "tests/", but it can be changed based on tests, for instance,
        # vllm sample test happens in samples/
        "vllm_basic_correctness_test": {
            "title": "Basic Correctness Test",
            "id": "vllm_basic_correctness_test",
            "env_var": {
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            },
            # test step:
            # required: command
            # available fields: env_var (env_var only set within the scope of the test step), package_install(pip package)
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


def validate_cuda(value: str) -> bool:
    VALID_VALUES = {"8.0", "8.9", "9.0"}
    return all(v in VALID_VALUES for v in value.split())


def check_versions():
    """
    check installed packages version
    """
    logger.info("Double check installed packages")
    patterns = ["torch", "xformers", "torchvision", "torchaudio", "vllm"]
    for pkg in patterns:
        pkg_exists(pkg)
    logger.info("Done. checked installed packages")


def preprocess_test_in(
    target_file: str = "requirements/test.in", additional_packages: Iterable[str] = ()
):
    """
    This modifies the target_file file in place in vllm work directory.
    It removes torch and unwanted packages in target_file and replace with local torch whls
    package  with format "$WHEEL_PACKAGE_NAME @ file://<LOCAL_PATH>"
    """
    additional_package_to_move = list(additional_packages or ())
    pkgs_to_remove = [
        "torch",
        "torchvision",
        "torchaudio",
        "xformers",
        "mamba_ssm",
    ] + additional_package_to_move
    # Read current requirements
    target_path = Path(target_file)
    lines = target_path.read_text().splitlines()

    # Remove lines starting with the package names (==, @, >=) — case-insensitive
    pattern = re.compile(rf"^({'|'.join(pkgs_to_remove)})\s*(==|@|>=)", re.IGNORECASE)
    kept_lines = [line for line in lines if not pattern.match(line)]

    # Get local installed torch/vision/audio from pip freeze
    # This is hacky, but it works
    pip_freeze = subprocess.check_output(["pip", "freeze"], text=True)
    header_lines = [
        line
        for line in pip_freeze.splitlines()
        if re.match(
            r"^(torch|torchvision|torchaudio)\s*@\s*file://", line, re.IGNORECASE
        )
    ]

    # Write back: header_lines + blank + kept_lines
    out = "\n".join(header_lines + [""] + kept_lines) + "\n"
    target_path.write_text(out)
    logger.info("[INFO] Updated %s", target_file)

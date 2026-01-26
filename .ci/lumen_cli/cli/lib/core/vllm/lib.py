import logging
import os
import textwrap
from pathlib import Path
from typing import Any

import yaml
from cli.lib.common.gh_summary import write_gh_step_summary
from cli.lib.common.git_helper import clone_external_repo
from cli.lib.common.pip_helper import pip_install_packages
from cli.lib.common.utils import run_command, temp_environ, working_directory
from jinja2 import Template


VLLM_DEFAULT_RERUN_FAILURES_COUNT = 2
VLLM_DEFAULT_RERUN_FAILURES_DELAY = 10


logger = logging.getLogger(__name__)

_VLLM_TEST_LIBRARY_PATH = Path(__file__).parent / "vllm_test_library.yaml"


def _load_vllm_test_library_yaml() -> dict[str, Any]:
    """
    Load the VLLM test library configuration from YAML file.

    Returns:
        Dictionary containing the test library configuration.
    """
    if not _VLLM_TEST_LIBRARY_PATH.exists():
        raise FileNotFoundError(
            f"VLLM test library YAML file not found: {_VLLM_TEST_LIBRARY_PATH}"
        )

    with open(_VLLM_TEST_LIBRARY_PATH, encoding="utf-8") as f:
        _vllm_test_library_cache = yaml.safe_load(f)

    return _vllm_test_library_cache


_TPL_VLLM_INFO = Template(
    textwrap.dedent("""\
    ##  Vllm against Pytorch CI Test Summary
    **Vllm Commit**: [{{ vllm_commit }}](https://github.com/vllm-project/vllm/commit/{{ vllm_commit }})
    {%- if torch_sha %}
    **Pytorch Commit**: [{{ torch_sha }}](https://github.com/pytorch/pytorch/commit/{{ torch_sha }})
    {%- endif %}
""")
)


def sample_vllm_test_library() -> dict[str, Any]:
    """
    Load the VLLM test library configuration from YAML file.

    This is a simple sample to unblock the vllm ci development, which mimics
    https://github.com/vllm-project/vllm/blob/main/.buildkite/test-pipeline.yaml
    See run_test_plan for more details.

    Returns:
        Dictionary containing test configurations loaded from vllm_test_library.yaml
    """
    return _load_vllm_test_library_yaml()


def check_parallelism(tests: Any, title: str, shard_id: int = 0, num_shards: int = 0):
    """
    a method to check if the test plan is parallelism or not.
    """
    parallelism = int(tests.get("parallelism", "0"))
    is_parallel = parallelism and parallelism > 1

    if not is_parallel:
        return False

    if shard_id > num_shards:
        raise RuntimeError(
            f"Test {title} expects {num_shards} shards, but invalid {shard_id} is provided"
        )

    if num_shards != parallelism:
        raise RuntimeError(
            f"Test {title} expects {parallelism} shards, but invalid {num_shards} is provided"
        )

    return True


def run_test_plan(
    test_plan: str,
    test_target: str,
    tests_map: dict[str, Any],
    shard_id: int = 0,
    num_shards: int = 0,
):
    """
    a method to run list of tests based on the test plan.
    """
    logger.info("run %s tests.....", test_target)
    if test_plan not in tests_map:
        raise RuntimeError(
            f"test {test_plan} not found, please add it to test plan pool"
        )
    tests = tests_map[test_plan]
    pkgs = tests.get("package_install", [])
    title = tests.get("title", "unknown test")

    is_parallel = check_parallelism(tests, title, shard_id, num_shards)
    if is_parallel:
        title = title.replace("%N", f"{shard_id}/{num_shards}")

    logger.info("Running tests: %s", title)
    if pkgs:
        logger.info("Installing packages: %s", pkgs)
        pip_install_packages(packages=pkgs, prefer_uv=True)
    with (
        working_directory(tests.get("working_directory", "tests")),
        temp_environ(tests.get("env_vars", {})),
    ):
        failures = []
        for step in tests["steps"]:
            logger.info("Running step: %s", step)
            if is_parallel:
                step = replace_buildkite_placeholders(step, shard_id, num_shards)
                logger.info("Running parallel step: %s", step)
            # Support retry with delay for all pytest commands, pytest-rerunfailures
            # is already a dependency of vLLM. This is needed as a stop gap to reduce
            # the number of requests to HF until #172300 can be landed to enable
            # HF offline mode
            if step.startswith("pytest"):
                # Use a low retry count and a high delay value to lower the risk of
                # having a retry storm and make thing worse
                rerun_count = os.getenv(
                    "VLLM_RERUN_FAILURES_COUNT", VLLM_DEFAULT_RERUN_FAILURES_COUNT
                )
                rerun_delay = os.getenv(
                    "VLLM_RERUN_FAILURES_DELAY", VLLM_DEFAULT_RERUN_FAILURES_DELAY
                )
                if rerun_delay:
                    step = step.replace(
                        "pytest",
                        f"pytest --reruns {rerun_count} --reruns-delay {rerun_delay}",
                        1,
                    )
                else:
                    step = step.replace(
                        "pytest",
                        f"pytest --reruns {rerun_count}",
                        1,
                    )

            code = run_command(cmd=step, check=False, use_shell=True)
            if code != 0:
                failures.append(step)
            logger.info("Finish running step: %s", step)
        if failures:
            logger.error("Failed tests: %s", failures)
            raise RuntimeError(f"{len(failures)} pytest runs failed: {failures}")
        logger.info("Done. All tests passed")


def clone_vllm(dst: str = "vllm"):
    _, commit = clone_external_repo(
        target="vllm",
        repo="https://github.com/vllm-project/vllm.git",
        dst=dst,
        update_submodules=True,
    )
    return commit


def replace_buildkite_placeholders(step: str, shard_id: int, num_shards: int) -> str:
    mapping = {
        "$$BUILDKITE_PARALLEL_JOB_COUNT": str(num_shards),
        "$$BUILDKITE_PARALLEL_JOB": str(shard_id),
    }
    for k in sorted(mapping, key=len, reverse=True):
        step = step.replace(k, mapping[k])
    return step


def summarize_build_info(vllm_commit: str) -> bool:
    torch_sha = os.getenv("GITHUB_SHA")
    md = (
        _TPL_VLLM_INFO.render(vllm_commit=vllm_commit, torch_sha=torch_sha).strip()
        + "\n"
    )
    return write_gh_step_summary(md)

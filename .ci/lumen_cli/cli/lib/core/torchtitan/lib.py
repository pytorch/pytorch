import logging
import os
from pathlib import Path
from typing import Any

import yaml
from cli.lib.common.git_helper import clone_external_repo
from cli.lib.common.utils import run_command, temp_environ, working_directory


logger = logging.getLogger(__name__)

_TORCHTITAN_TEST_LIBRARY_PATH = Path(__file__).parent / "torchtitan_test_library.yaml"


def _load_torchtitan_test_library_yaml() -> dict[str, Any]:
    if not _TORCHTITAN_TEST_LIBRARY_PATH.exists():
        raise FileNotFoundError(
            f"torchtitan test library YAML not found: {_TORCHTITAN_TEST_LIBRARY_PATH}"
        )
    with open(_TORCHTITAN_TEST_LIBRARY_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_torchtitan_test_library() -> dict[str, Any]:
    return _load_torchtitan_test_library_yaml()


def clone_torchtitan(dst: str = "torchtitan"):
    _, commit = clone_external_repo(
        target="torchtitan",
        repo="https://github.com/pytorch/torchtitan.git",
        dst=dst,
    )
    return commit


def run_test_plan(
    test_plan: str,
    tests_map: dict[str, Any],
):
    logger.info("Running torchtitan test plan: %s", test_plan)
    if test_plan not in tests_map:
        raise RuntimeError(
            f"test plan '{test_plan}' not found in torchtitan test library"
        )

    tests = tests_map[test_plan]
    title = tests.get("title", "unknown test")
    logger.info("Running tests: %s", title)

    # torchtitan's tests/integration_tests/features.py builds the
    # --checkpoint.initial_load_path as $RUNNER_TEMP/artifacts-to-be-uploaded/...
    # when RUNNER_TEMP is set, but the paired save reads OUTPUT_DIR (default
    # "artifacts-to-be-uploaded", relative to cwd). Pin OUTPUT_DIR to the same
    # absolute prefix so save and load land in the same directory.
    env_vars = dict(tests.get("env_vars", {}))
    runner_temp = os.environ.get("RUNNER_TEMP")
    if runner_temp:
        env_vars.setdefault(
            "OUTPUT_DIR", os.path.join(runner_temp, "artifacts-to-be-uploaded")
        )

    with (
        working_directory(tests.get("working_directory", "")),
        temp_environ(env_vars),
    ):
        failures = []
        for step in tests["steps"]:
            logger.info("Running step: %s", step)
            code = run_command(cmd=step, check=False, use_shell=True)
            if code != 0:
                failures.append(step)
            logger.info("Finished step: %s", step)
        if failures:
            logger.error("Failed steps: %s", failures)
            raise RuntimeError(f"{len(failures)} test steps failed: {failures}")
        logger.info("All tests passed for plan: %s", test_plan)

import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List

from cli.lib.common.file_utils import read_yaml_file
from cli.lib.common.utils import (
    list_to_env_dict,
    run_bash_and_capture_env,
    run_cmd,
    temp_environ,
    working_directory,
)


logger = logging.getLogger(__name__)

@dataclass
class TestPlan:
    """
    Configuration for test plan.
    """
    name: str
    id: str
    env_vars: List[str] = field(default_factory=list)
    env_bash: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    test_directory: str = ""  # by default,use default test directory

@dataclass
class TestConfig:
    """
    Configuration for test plan.
    """
    name: str
    test_plans: dict[str, TestPlan]
    preset: list[Any] | None = None
    env_vars: list[str] | None = None
    env_bash: str = ""
    work_directory: str = "."  # by default, use the current working directory which is root of the torch repo
    default_test_directory: str = "tests"  # by default, use the test directory in the torch repo, can be overridden by test plans


class TestRunner(ABC):
    """
    Base class for defining a test runner.

    Users should subclass this and implement the `run()` methods.
    """

    def __init__(self, config_path: str = "", test_ids: list[str] = []):
        """
        Initialize the build runner.

        Args:
            config_path (str): Optional path to a config file (YAML, JSON, etc).
        """
        self.config_path = config_path
        self.test_config = self.to_test_config_model()
        self.test_plan_ids = self.validate_input_ids(test_ids, self.test_config)

    def load_raw_test_config(self):
        """
        load config from file.
        Override this if you want custom parsing logic.
        """
        config = {}
        if self.config_path:
            logger.info("use config file user provided")
            logger.info("Reading config yaml file ...")
            config = read_yaml_file(self.config_path)
        else:
            logger.info("did not find the config file, use default behaviour")
        return config.get("test", {})

    def to_test_config_model(self):
        raw = self.load_raw_test_config()
        required_top_keys = {"test_plans"}
        missing = required_top_keys - raw.keys()
        if missing:
            raise ValueError(f"Missing required keys in TestConfig: {missing}")

        logger.info(f"parsing test config: {raw}")

        # --- Validate and construct test_plans ---
        plans = {}
        for i, plan in enumerate(raw["test_plans"], start=1):
            if not isinstance(plan, dict):
                raise TypeError(f"test_plans[{i}] must be a dict, got {type(plan)}")
            for key in ("name", "id", "steps"):
                if key not in plan:
                    raise ValueError(f"test_plans[{i}] missing required key '{key}'")
            if plan["id"] in plans:
                raise ValueError(
                    f"Duplicate test id '{plan['id']}', please use unique ids in test plans"
                )
            plans[plan["id"]] = TestPlan(**plan)

        return TestConfig(
            name=raw["name"],
            test_plans=plans,
            work_directory=raw.get("work_directory", "."),
            default_test_directory=raw.get("default_test_directory", "tests"),
            preset=raw.get("preset", []),
            env_vars=raw.get("env_vars", []),
        )

    def validate_input_ids(self, ids: list[str], test_config: TestConfig):
        test_plans = test_config.test_plans
        invalid_ids = [id for id in ids if id not in test_plans]
        if len(invalid_ids) > 0:
            raise ValueError(f"Invalid test ids: {invalid_ids}")
        return ids

    def prepare(self):
        """
        Prepare the test environment. this is used to set up the environment before running the tests
        """
        pass

    def run(self):
        self.prepare()
        logger.info(
            f"running test plans in sequence: [{', '.join(self.test_plan_ids)}]..."
        )
        config = self.test_config
        wd = config.work_directory if config.work_directory else "."
        logger.info(f"working directory: {wd}")
        with temp_environ(get_envs(config.env_vars, config.env_bash)), working_directory(wd):
            for id in self.test_plan_ids:
                test_plan = self.test_config.test_plans[id]
                self.run_test_plan(test_plan)

    def run_test_plan(self, test_plan: TestPlan):
        """
        Run the test plan.
        """
        with temp_environ(get_envs(test_plan.env_vars, test_plan.env_bash)):
            logger.info(f"Running test plan: {test_plan.name}...")
            for step in test_plan.steps:
                self.run_test_step(step)

    def run_test_step(self, step: dict[str, Any]):
        """
        Run the test step.
        """
        command = step.get("step", "")
        working_directory = step.get("working_directory", "")
        logger.info(f"Running test step command: {command}...")
        with temp_environ(list_to_env_dict(step.get("env_vars", []))):
            run_cmd(command)


def get_envs(env_vars: list[str] | None = None, env_bash: str = "") -> Dict[str, str]:
    envs, bash_envs = {}, {}
    if env_vars:
        envs = list_to_env_dict(env_vars)
    if env_bash:
        bash_envs = run_bash_and_capture_env(env_bash)
    merged_env = {**envs, **bash_envs}
    return merged_env

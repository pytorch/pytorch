import glob
import os
import shlex
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

from lib.utils import clone_vllm, get_post_build_pinned_commit, read_yaml_file, run
import os

class VllmTestRunner:
    def __init__(self, file_path="") -> None:
        self.test_configs = self._fetch_configs(file_path)

    def run(self, test_names):
        valid_tests = []
        for test_name in test_names:
            if test_name not in self.test_configs:
                print(
                    f"[warning] cannot detect test name {test_name}, please input valid test name "
                )
                continue
            config = self.test_configs.get(test_name)
            valid_tests.append(config)
        os.chdir("vllm")
        for config in valid_tests:
            self.test(config)
        os.chdir("..")

    def test(self, config={}):
        testid = config["id"]
        steps = config["steps"]
        sub_path = config.get("path", ".")
        print(f"running test config: {testid}")
        for step in steps:
            run(step, cwd=sub_path, logging=True)

    def _fetch_configs(self, path=""):
        base_dir = os.path.dirname(__file__)
        file_path = path if path else os.path.join(base_dir, "test_config.yaml")
        res = read_yaml_file(file_path)
        config_map = {}
        for item in res:
            if "id" in item:
                config_map[item["id"]] = item
            else:
                raise ValueError(f"Missing 'id' in config: {item}")

        print(f"config_map: {config_map}")
        return config_map

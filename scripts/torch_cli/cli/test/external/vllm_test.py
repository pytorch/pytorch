import os
from cli.lib.utils import read_yaml_file, run_shell
import os
import sys

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
        sub_path = config.get("path", "tests")
        print(f"running test config: {testid}")
        envs = os.environ.copy()
        if os.environ.get("VLLM_TEST_HUGGING_FACE_TOKEN", ""):
            print("found VLLM_TEST_HUGGING_FACE_TOKEN in env",flush=True)
            sys.stdout.flush()
        else:
            print("VLLM_TEST_HUGGING_FACE_TOKEN not found in env",flush=True)
            sys.stdout.flush()

        envs["HF_TOKEN"] = os.environ.get("VLLM_TEST_HUGGING_FACE_TOKEN", "")
        for step in steps:
            # todo : replace with run_cmd with envrirnment
            run_shell(step, cwd=sub_path, env=envs)

    def _fetch_configs(self, path=""):
        base_dir = os.path.dirname(__file__)
        file_path = path if path else os.path.join(base_dir, "vllm_test_config.yaml")
        res = read_yaml_file(file_path)
        config_map = {}
        for item in res:
            if "id" in item:
                config_map[item["id"]] = item
            else:
                raise ValueError(f"Missing 'id' in config: {item}")

        print(f"config_map: {config_map}")
        return config_map

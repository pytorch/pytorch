"""
Generate a torchbench test report from a file containing the PR body.
Currently, only supports running tests on specified model names

Testing environment:
- Intel Xeon 8259CL @ 2.50 GHz, 24 Cores with disabled Turbo and HT
- Nvidia Tesla T4
- Nvidia Driver 450.51.06
- Python 3.7
- CUDA 10.2
"""
# Known issues:
# 1. Does not reuse the build artifact in other CI workflows
# 2. CI jobs are serialized because there is only one worker
import os
import git
import enum
import json
import argparse
import subprocess

CUDA_VERSION = "cu102"
PYTHON_VERSION = "3.7"
TORCHBENCH_CONFIG_NAME = "config.yaml"
TORCHBENCH_RESULT_NAME = "result.txt"
PYTORCH_SRC = os.path.join(os.environ("HOME"), "pytorch")
ABTEST_CONFIG_TEMPLATE = """
start: {control}
end: {treatment}
threshold: 100
direction: decrease
timeout: 60
tests:"""

def gen_abtest_config(control: str, treatment: str, tests: str, models: List[str]):
    d = {}
    d["control"] = control
    d["treatment"] = treatment
    config = ABTEST_CONFIG_TEMPLATE.format(**d)
    for model in models:
        config = f"{config}\n  - {model}"
    config = config + "\n"
    return config

def deploy_torchbench_config(config: str):
    # TorchBench config file name
    config_path = os.path.join(OUTPUT_DIR, TORCHBENCH_CONFIG_NAME)
    with open(config_path, "w") as fp:
        fp.write(config)

def extract_models_from_pr(torchbench_path: str, prbody_file: str) -> List[str]:
    model_list = []
    MAGIC_PREFIX = "RUN-TORCHBENCH:"
    with open(prbody_file, "r") as pf:
        magic_lines = filter(lambda x: x.startswith(MAGIC_PREFIX), pf.lines())
        if magic_lines:
            model_list = list(map(lambda x: x.strip(), magic_lines[0][len(MAGIC_PREFIX)].split(",")))
    # Sanity check: make sure all the user specified models exist in torchbench repository
    full_model_list = os.listdir(os.path.join(torchbench_path, "torchbenchmark", "models"))
    for m in model_list:
        if not m in full_model_list:
            print(f"The model {m} you specified does not exist in TorchBench suite. Please double check.")
            return []
    return model_list

def run_torchbench(torchbench_path: str, pr_num: str, out_dir: str, conda_env: str):
    # Copy system environment so that we will not override
    env = dict(os.environ)
    # Always rebuild
    env["BISECT_CONDA_ENV"] = conda_env
    env["BISECT_ISSUE"] = f"pr{pr_num}"
    command = ["bash", "./.github/scripts/run-bisection.sh"]
    subprocess.check_call(command, cwd=torchbench_path, env=env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TorchBench tests based on PR')
    parser.add_argument('pr-num', required=True, type=str, help="The Pull Request number")
    parser.add_argument('pr-base-sha', required=True, type=str, help="The Pull Request base hash")
    parser.add_argument('pr-head-sha', required=True, type=str, help="The Pull Request head hash")
    parser.add_argument('pr-body', required=True, type=argparse.FileType('r'),
                        help="The file that contains body of a Pull Request")
    parser.add_argument('conda-env', required=True, type=str, help="Name of the conda env to run the test")
    parser.add_argument('torchbench-path', required=True, type=str, help="Path to TorchBench repository")
    args = parser.parse_args()
    
    output_dir: str = os.path.join(os.environ("HOME"), ".torchbench", "bisection", f"pr{}")
    # Identify the specified models and verify the input
    models = extract_models_from_pr(args.torchbench_path, args.pr_body)
    if not models:
        print(f"Can't parse the model filter from the pr body. Currently we only support allow-list.")
        return
    print(f"Ready to run TorchBench with benchmark. Result will be saved in the directory: {OUTPUT_DIR}.")
    # Run TorchBench with the generated config
    torchbench_config = gen_abtest_config(args.pr_base_sha, args.pr_head_sha, models)
    deploy_torchbench_config(torchbench_config)
    run_torchbench(repos["benchmark"], pr_num = args.pr_num, out_dir = output_dir, conda_env = args.conda_env)

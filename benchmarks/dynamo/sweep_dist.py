import argparse
import os
import re
import subprocess
import sys

from tabulate import tabulate

model_args = [
    ["--toy_model"],
    ["--torchbench_model", "hf_Bert"],
    ["--torchbench_model", "hf_T5"],
    ["--torchbench_model", "resnet50"],
    ["--torchbench_model", "hf_T5_large"],
]
dynamo_args = [
    [],
    ["--dynamo", "aot_eager"],
    ["--dynamo", "inductor"],
]


def parse_log(log_lines):
    log_str = "".join(log_lines)
    code = re.findall(r"PASS|FAIL", log_str)
    errors = re.findall(r".*Error:.*", log_str)
    if len(code) < 1:
        code = "ERROR"
    else:
        assert len(code) == 1
        code = code[0]
    return {
        "code": code,
        "errors": errors,
    }


TRUNCATE = 90


def sweep_fsdp(script_file="benchmarks/dist_bench.py"):
    disable_fake_tensor = ["--disable_fake_tensor"]
    fsdp = ["--fsdp"]
    args_base = [script_file] + fsdp + disable_fake_tensor
    runs = []
    for model in model_args:
        for dynamo in dynamo_args:
            args = args_base + model + dynamo
            model_name = "toy_model" if len(model) == 1 else model[1]
            backend_name = "eager" if len(dynamo) == 0 else dynamo[1]
            run_str = f"{model_name}_{backend_name}"
            try:
                proc = subprocess.Popen(
                    [sys.executable] + args,
                    encoding="utf-8",
                    stderr=subprocess.STDOUT,
                    stdout=subprocess.PIPE,
                )
                proc.wait()
                out, _ = proc.communicate()
                with open(f"dist_logs/{run_str}.log", "w") as f:
                    f.write(out)
                errors = re.findall(r".*Error:.*", out)
                if len(errors):
                    code = "FAIL"
                    err = errors[-1]
                    print(run_str)
                    print(out)
                else:
                    code = "PASS"
                    err = ""
            except subprocess.SubprocessError as e:
                with open(f"dist_logs/{run_str}.log", "w+") as f:
                    f.write(f"ERROR, {e}")
                print(out)
                code = "FAIL"
                err = str(e)
            print(f"{run_str}:  {code}, {err}")
            runs.append((model_name, backend_name, code, err[:TRUNCATE]))

    print(tabulate(runs, headers=("Model", "Backend", "Code", "Error")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsdp")
    sweep_fsdp()

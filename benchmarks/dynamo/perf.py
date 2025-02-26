from common import parse_args, run
import torch
from torchbench import setup_torchbench_cwd, TorchBenchmarkRunner
import torch._inductor.config as ic
import os
import sys

from contextlib import redirect_stdout
import io
import re

stdout_capture = io.StringIO()


def run_model(model):
    original_dir = setup_torchbench_cwd()
    try:
        args = parse_args(
            [
                "--inductor",
                "--training",
                "--performance",
                f"--only={model}",
            ]
        )
        run(TorchBenchmarkRunner(), args, original_dir)
    finally:
        os.chdir(original_dir)



def speedup(model):
    # with capture_stdout() as output:
    #     run_model("basic_gnn_sage")

    with redirect_stdout(stdout_capture):
        run_model(model)
    # Get the captured output
    captured_output = stdout_capture.getvalue()
    mch = re.search(r'(\d+(\.\d+)?)x', captured_output)

    return mch.group(0)

if __name__ == "__main__":
    ic.force_disable_caches = True
    before = speedup("basic_gnn_gcn")
    ic.max_fusion_size = 1
    torch._dynamo.reset()
    after = speedup("basic_gnn_gcn")
    print(before + " " + after)

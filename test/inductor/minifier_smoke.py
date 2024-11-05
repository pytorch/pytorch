# Owner(s): ["module: inductor"]
#
# This smoketest is referenced in the internal-only minifier runbook
# https://docs.google.com/document/d/18L9e7bZSBpJ7gGbwlUV13LasmjiEX2lree2pl-SdbCU/edit
import os


os.environ["TORCHDYNAMO_REPRO_AFTER"] = "dynamo"
import torch
import torch._dynamo as torchdynamo
import torch._inductor.config
import torch._ops


torch._inductor.config.cpp.inject_relu_bug_TESTING_ONLY = "compile_error"


def func(x):
    x = torch.sigmoid(x)
    x = torch.mul(x, torch.ones(2))
    x = torch.relu(x)
    x = torch.add(x, torch.zeros(2))
    x = torch.ops.aten.round(x)
    return x


def run_internal_minifier():
    torchdynamo.config.debug_dir_root = "."
    f_opt = torch.compile(func)
    f_opt(torch.ones(2))


run_internal_minifier()

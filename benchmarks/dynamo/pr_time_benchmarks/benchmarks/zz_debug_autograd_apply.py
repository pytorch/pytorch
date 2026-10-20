"""DEBUG ONLY - do not land. Diagnoses why autograd_function_apply costs ~3.9k
more instructions per call in CI on from-source builds of this stack.

Prints diagnostics only; appends no benchmark results. Each measurement runs in a
fresh subprocess so import-time state matches the real benchmark process.
"""

import json
import os
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
PRE_CONFIG = os.path.join(os.path.dirname(HERE), "debug_config_module_pre.py")

# Runs in the child. argv: config(cur|pre) pad_strings junk_types
CHILD = r"""
import importlib.abc, importlib.util, json, sys

config, pad, types_ = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

if config == "pre":
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "torch.utils._config_module":
                return importlib.util.spec_from_file_location(name, sys.argv[4])
    sys.meta_path.insert(0, Finder())

keep = ["x" * (i % 97 + 1) + str(i) for i in range(pad)]
junk = [type(f"J{i}", (), {"v": i}) for i in range(types_)]
for c in junk:
    c.v

import torch
import torch._C._instruction_counter as ic


class NoSave(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *ps):
        return OUT

    @staticmethod
    def backward(ctx, *g):
        raise RuntimeError


def count(fn, n=7):
    for _ in range(3):
        fn()
    best = None
    for _ in range(n):
        i = ic.start()
        fn()
        c = ic.end(i)
        best = c if best is None else min(best, c)
    return best


res = {"config_module": torch.utils._config_module.__file__}
for dev in ("cuda", "cpu"):
    OUT = (torch.empty(0, device=dev), torch.empty(0, device=dev))
    args = tuple(torch.empty(0, device=dev, requires_grad=True) for _ in range(13))
    noreq = tuple(torch.empty(0, device=dev) for _ in range(13))
    res[f"apply_{dev}"] = count(lambda: NoSave.apply(*args))
    res[f"apply_{dev}_noreqgrad"] = count(lambda: NoSave.apply(*noreq))
OUT = (torch.empty(0, device="cuda"), torch.empty(0, device="cuda"))
args = tuple(torch.empty(0, device="cuda", requires_grad=True) for _ in range(13))
for attr in ("_backward_cls", "forward", "clear_saved_tensors_on_access",
             "boxed_grads_call", "setup_context", "apply"):
    res[f"getattr_{attr}"] = count(lambda: getattr(NoSave, attr))
res["backward_cls_call"] = count(lambda: NoSave._backward_cls())
res["empty_call"] = count(lambda: None)
sys._clear_type_cache()
res["apply_cuda_after_clear_type_cache"] = count(lambda: NoSave.apply(*args))

calls = []
def prof(frame, event, arg):
    if event in ("call", "c_call"):
        calls.append(frame.f_code.co_qualname if event == "call" else getattr(arg, "__qualname__", "?"))
sys.setprofile(prof)
NoSave.apply(*args)
sys.setprofile(None)
res["python_calls_in_apply"] = calls
res["state"] = {
    "tf_mode_stack": torch._C._len_torch_function_stack(),
    "functorch_active": torch._C._are_functorch_transforms_active(),
    "anomaly": torch.is_anomaly_enabled(),
    "profiler": torch.autograd.profiler._is_profiler_enabled,
    "saved_tensor_hooks": torch._C._autograd._top_saved_tensors_default_hooks(False) is not None,
    "grad_enabled": torch.is_grad_enabled(),
    "python": sys.version,
    "modules_loaded": len(sys.modules),
}
print("DEBUGJSON " + json.dumps(res))
"""


def run(config, pad=0, types_=0):
    out = subprocess.run(
        [sys.executable, "-c", CHILD, config, str(pad), str(types_), PRE_CONFIG],
        capture_output=True,
        text=True,
    )
    for line in out.stdout.splitlines():
        if line.startswith("DEBUGJSON "):
            return json.loads(line[len("DEBUGJSON ") :])
    return {"error": out.stderr[-2000:]}


def main():
    print("=== DEBUG autograd_function_apply diagnostics")
    for config in ("cur", "pre"):
        for rep in range(2):
            print(f"DEBUG config={config} rep={rep}: {json.dumps(run(config))}")
    for pad in (0, 5000, 50000):
        for types_ in (0, 17, 200):
            r = run("cur", pad, types_)
            print(
                f"DEBUG sweep pad={pad} types={types_}: apply_cuda={r.get('apply_cuda')} apply_cpu={r.get('apply_cpu')}"
            )
    print("=== DEBUG done")


if __name__ == "__main__":
    main()

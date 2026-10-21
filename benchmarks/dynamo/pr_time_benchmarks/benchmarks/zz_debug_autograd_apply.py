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

HOOK = r"""
import importlib.abc, importlib.util, json, os, sys
config = sys.argv[1]

if config == "pre":
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "torch.utils._config_module":
                return importlib.util.spec_from_file_location(name, os.environ["DEBUG_PRE_CONFIG"])
    sys.meta_path.insert(0, Finder())
"""

# Runs in the child. argv: config(cur|pre) pad_strings junk_types - dyn warm named benchmarks_dir
CHILD = (
    HOOK
    + r"""
pad, types_ = int(sys.argv[2]), int(sys.argv[3])
dyn, warm, named = (int(a) for a in sys.argv[5:8])
keep = ["x" * (i % 97 + 1) + str(i) for i in range(pad)]
junk = [type(f"J{i}", (), {"v": i}) for i in range(types_)]
for c in junk:
    c.v

if dyn:
    sys.path.insert(0, sys.argv[8])
    import benchmark_base
import torch
import torch._C._instruction_counter as ic


class NoSave(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *ps):
        return OUT


class Named(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13):
        return OUT

    @staticmethod
    def backward(ctx, *g):
        raise RuntimeError


if named:
    NoSave = Named

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


res = {"torch": torch.__file__, "config_module": torch.utils._config_module.__file__}
OUT = (torch.empty(0, device="cuda"), torch.empty(0, device="cuda"))
args = tuple(torch.empty(0, device="cuda", requires_grad=True) for _ in range(13))
for _ in range(10000 * warm):
    NoSave.apply(*args)
res["apply_cuda_first"] = count(lambda: NoSave.apply(*args))
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
)


def run(config, pad=0, types_=0, dyn=0, warm=0, named=0):
    out = subprocess.run(
        # -P: CI runs from the repo root, whose torch/ would shadow the installed wheel.
        [sys.executable, "-P", "-c", CHILD, config, str(pad), str(types_), "-"]
        + [str(dyn), str(warm), str(named), HERE],
        capture_output=True,
        text=True,
        env={**os.environ, "DEBUG_PRE_CONFIG": PRE_CONFIG},
    )
    for line in out.stdout.splitlines():
        if line.startswith("DEBUGJSON "):
            return json.loads(line[len("DEBUGJSON ") :])
    return {"error": out.stderr[-2000:]}


# The real benchmark, run in a child under either config
REAL = (
    HOOK
    + r"""
import runpy
sys.path.insert(0, sys.argv[2])
sys.argv = [sys.argv[3], sys.argv[4]]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
)


def run_real(config):
    script = os.path.join(HERE, "autograd_function.py")
    out = subprocess.run(
        [sys.executable, "-P", "-c", REAL, config, HERE, script, os.devnull],
        capture_output=True,
        text=True,
        env={**os.environ, "DEBUG_PRE_CONFIG": PRE_CONFIG},
    )
    counts = [l.split()[-1] for l in out.stdout.splitlines() if "for iteration" in l]
    return counts or out.stderr[-2000:]


# The real benchmark's work, measured after each of several pre-iteration actions
PROBE = (
    HOOK
    + r"""
import gc, runpy
import torch

# Each setattr on a type invalidates its version tag, and the lookup reassigns a new
# one, so this shifts the tags (and method-cache slots) of every type tagged later.
Burn = type("Burn", (), {})
for k in range(int(sys.argv[4])):
    Burn.x = k
    Burn.x
sys.path.insert(0, sys.argv[2])
g = runpy.run_path(sys.argv[3], run_name="dbg")
import torch._C._instruction_counter as ic
from sympy.core.cache import clear_cache

b = g["Benchmark"]()
b._prepare_once()


def rewarm():
    clear_cache()
    b._work()


res = {}
for name, pre in [
    ("none", lambda: None),
    ("sympy_clear", clear_cache),
    ("type_cache_clear", sys._clear_type_cache),
    ("sympy_clear_then_work", rewarm),
    ("gc_collect", gc.collect),
    ("sympy_clear2", clear_cache),
    ("none2", lambda: None),
]:
    counts = []
    for _ in range(5):
        pre()
        i = ic.start()
        b._work()
        counts.append(ic.end(i))
    res[name] = counts
print("DEBUGJSON " + json.dumps(res))
"""
)


def run_probe(config, burn=0):
    script = os.path.join(HERE, "autograd_function.py")
    out = subprocess.run(
        [sys.executable, "-P", "-c", PROBE, config, HERE, script, str(burn)],
        capture_output=True,
        text=True,
        env={**os.environ, "DEBUG_PRE_CONFIG": PRE_CONFIG},
    )
    for line in out.stdout.splitlines():
        if line.startswith("DEBUGJSON "):
            return line[len("DEBUGJSON ") :]
    return out.stderr[-2000:]


def main():
    print("=== DEBUG autograd_function_apply diagnostics")
    for config in ("cur", "pre"):
        for rep in range(2):
            print(f"DEBUG probe config={config} rep={rep}: {run_probe(config)}")
    for config in ("cur", "pre"):
        for burn in (1, 2, 3, 5, 8, 13, 50, 200, 950, 3000):
            r = run_probe(config, burn)
            try:
                r = {k: min(v) for k, v in json.loads(r).items()}
            except ValueError:
                pass
            print(f"DEBUG burn config={config} burn={burn}: {r}")
    for config in ("cur", "pre"):
        for rep in range(2):
            print(f"DEBUG real config={config} rep={rep}: {run_real(config)}")
    print("=== DEBUG done")


if __name__ == "__main__":
    main()

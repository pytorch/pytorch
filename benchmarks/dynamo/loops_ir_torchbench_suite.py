"""Genuine ten-model TorchBench suite for the experimental C++ LoopsIR lane.

Primary suite (exact source defaults, explicit ``--precision fp32`` for every
variant so device-metadata overrides cannot silently change the workload)::

    resnet18               train16 / eval8
    resnet50               train32 / eval32
    alexnet                train128 / eval128
    vgg16                  train64 / eval4
    mobilenet_v2           train96 / eval16
    BERT_pytorch           train16 / eval16, S128
    hf_Bert                train4 / eval1, S512
    hf_GPT2                train4 / eval1, S512 train / S1024 eval
    timm_vision_transformer train32 / eval32 (vit_small_patch16_224)
    dlrm                   train2048 / eval2048

No shape/depth/dtype/optimizer/loss changes. Native public weights are used
cached (downloads of the usual public weights are normal model dependencies);
never a randomly-initialized replacement for a pretrained model.

Genuine driver: every variant is driven through the real
``BenchmarkModel.invoke()`` staged convention (vision/HF
forward-loss/backward/optimizer step; custom BERT NLL + scheduled optimizer;
Timm SGD + CrossEntropy; DLRM MSE + SGD + scheduler). The compiler wrapper
compiles the module returned by ``get_module()`` and reinstalls it with
``set_module()`` so Parameter identity and optimizer references are
preserved; AOT compiles backward while the optimizer step may stay eager
(identical original step, explicitly reported). No generic
``reduce_to_scalar_loss`` surrogate is ever invented. Training ``invoke()``
may return ``None``; the real forward/loss is captured with a temporary
stage/observer hook using actual native loss semantics (removed before
timing). If a loss is unavailable it is documented, never fabricated.

Per-graph CPP capture uses compile-time hooks only (a counting wrapper
around the production seam plus the existing ``_Collector`` /
``_current_collector`` scope, which today retains only the last graph).
Hooks never interpret IR at runtime and no Python callback payload enters
the IR. A training case only passes with evidence of a compiled backward
CPP graph -- forward-only is an explicit failure.

Controls: ``--load-only`` builds each factory, records inventory (module
path, param count, input shapes/dtypes/strides, batch/source, profiling
settings, incoming globals) with NO latency measurement. ``--accuracy-only``
skips timing. Failed runs are never timed as valid; skips / dependency /
OOM cases are never counted as passes. Scorecard JSON is rewritten
incrementally after every case (never truncated on a later failure).

Timing (when enabled): compile first, warm up, serialize GPU measurements,
rotate variant order, collect ``--repeats`` samples (default 7) with
``benchmark_gpu(is_vetted_benchmarking=True)``. Raw samples + median/min/max
and raw stock ratios with noise are reported; no fixed-percentage green
threshold and no deadline promise.

Run (isolated GPU; visibility is NOT hardcoded in Python)::

    CUDA_VISIBLE_DEVICES=6 <python> \
        benchmarks/dynamo/loops_ir_torchbench_suite.py \
        --models all --test both --cpp-numerics both \
        --repeats 7 --warmup 20 --output-dir DIR [--accuracy-only] [--load-only]

Compilation requires the native ``cpp.run_pipeline`` API. An absent or
incompatible native API is an error; there is no manual-chain fallback.
``--load-only`` inventory performs no compilation or GPU timing.
Typed graph contracts and native boundary proofs are checked before execution.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import importlib
import inspect
import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, NamedTuple

PYTORCH_ROOT = Path(__file__).resolve().parents[2]
if str(PYTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTORCH_ROOT))
BENCHMARK_ROOT = PYTORCH_ROOT / "benchmark"
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

# Reuse vetted helpers (never benchmarks/dynamo/common.py, which mutates
# globals on import).
try:
    from benchmarks.dynamo.loops_ir_benchmark import (  # type: ignore[import]
        _compare_outputs,
        _raw_bytes,
    )
except Exception:  # fallback only if run from an unexpected sys.path layout

    def _raw_bytes(tensor: Any) -> bytes:
        import torch as _torch

        return (
            tensor.detach()
            .contiguous()
            .reshape(-1)
            .view(_torch.uint8)
            .cpu()
            .numpy()
            .tobytes()
        )

    @dataclasses.dataclass
    class _CompareResultFallback:
        structures_equal: bool
        dtypes_equal: bool
        shapes_equal: bool
        bytewise_equal: bool
        max_abs_diff: float | None
        detail: str = ""

    def _compare_outputs(actual: Any, expected: Any, label: str):  # type: ignore[no-redef]
        import torch as _torch

        from torch.utils._pytree import tree_flatten as _tf

        try:
            fa, sa = _tf(actual)
            fe, se = _tf(expected)
        except Exception as error:
            return _CompareResultFallback(
                False, False, False, False, None, f"{label}: {error}"
            )
        if sa != se or len(fa) != len(fe):
            return _CompareResultFallback(
                True, False, False, False, None, f"{label}: structure"
            )
        for leaf in (*fa, *fe):
            if not isinstance(leaf, _torch.Tensor):
                return _CompareResultFallback(
                    True, False, False, False, None, f"{label}: non-tensor"
                )
        dt = all(a.dtype == b.dtype for a, b in zip(fa, fe))
        sh = all(
            tuple(a.shape) == tuple(b.shape) and tuple(a.stride()) == tuple(b.stride())
            for a, b in zip(fa, fe)
        )
        if not (dt and sh):
            return _CompareResultFallback(
                True, dt, sh, False, None, f"{label}: dtype/shape"
            )
        ok = True
        mx = 0.0
        for a, b in zip(fa, fe):
            if _raw_bytes(a) != _raw_bytes(b):
                ok = False
            try:
                mx = max(
                    mx,
                    float(
                        (a.detach().float().cpu() - b.detach().float().cpu())
                        .abs()
                        .max()
                        .item()
                    ),
                )
            except Exception:
                pass
        return _CompareResultFallback(
            True, True, True, ok, mx, f"{label}: {'EQ' if ok else 'MISMATCH'}"
        )


try:
    from benchmarks.dynamo.loops_ir_bert_benchmark import (  # type: ignore[import]
        numerical_settings,
    )
except Exception:

    def numerical_settings() -> dict[str, Any]:  # type: ignore[no-redef]
        import torch as _torch

        from torch._inductor import config as _cfg

        return {
            "deterministic_algorithms": _torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": _torch.backends.cudnn.deterministic,
            "cudnn_benchmark": _torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": _torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": _torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": _torch.get_float32_matmul_precision(),
            "emulate_precision_casts": _cfg.emulate_precision_casts,
            "disable_ftz": _cfg.eager_numerics.disable_ftz,
            "use_fast_math": _cfg.use_fast_math,
        }


EXTRA_ARGS = ["--precision", "fp32"]
DEVICE = "cuda"

# Exact source defaults (see module docstring). hf_GPT2 sequence length is
# 512 for train / 1024 for eval (basic_configs.py).
MODEL_MANIFEST: dict[str, dict[str, Any]] = {
    "resnet18": {"train_bs": 16, "eval_bs": 8},
    "resnet50": {"train_bs": 32, "eval_bs": 32},
    "alexnet": {"train_bs": 128, "eval_bs": 128},
    "vgg16": {"train_bs": 64, "eval_bs": 4},
    "mobilenet_v2": {"train_bs": 96, "eval_bs": 16},
    "BERT_pytorch": {"train_bs": 16, "eval_bs": 16, "seq_len": 128},
    "hf_Bert": {"train_bs": 4, "eval_bs": 1, "train_seq": 512, "eval_seq": 512},
    "hf_GPT2": {"train_bs": 4, "eval_bs": 1, "train_seq": 512, "eval_seq": 1024},
    "timm_vision_transformer": {
        "train_bs": 32,
        "eval_bs": 32,
        "variant": "vit_small_patch16_224",
    },
    "dlrm": {"train_bs": 2048, "eval_bs": 2048},
}
MODEL_CHOICES = sorted(MODEL_MANIFEST)

# Legacy audit tolerances (NOT acceptance; kept for audit continuity only).
# Normal acceptance uses the stock-effective comparator
# (benchmark/torchbenchmark/util/env_check.py:get_tolerance_and_cosine_flag
# + check_accuracy via torch/_dynamo/utils.py:same with a genuine FP64
# reference). The assert_close fields below stay as diagnostics.
NORMAL_RTOL = 1e-4
NORMAL_ATOL = 1e-5

# Explicit recorded sources for the stock-effective normal lane (no
# handwritten per-model overrides; policy comes from these APIs).
STOCK_TOLICY_SOURCE = (
    "benchmark/torchbenchmark/util/env_check.py:get_tolerance_and_cosine_flag"
)
STOCK_CHECK_SOURCE = "benchmark/torchbenchmark/util/env_check.py:check_accuracy"
STOCK_COMPARATOR_SOURCE = "torch/_dynamo/utils.py:same"
STOCK_FP64_SOURCE = (
    "benchmark/torchbenchmark/util/env_check.py:check_accuracy fp64 golden ref "
    "mirrored: separate deep-copied benchmark cast to float64 + same original "
    "driver; reference-only double, live workload never touched, RNG/global "
    "numerics restored"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=["all"], choices=[*MODEL_CHOICES, "all"]
    )
    parser.add_argument("--test", choices=("eval", "train", "both"), default="both")
    parser.add_argument(
        "--cpp-numerics", choices=("strict_eager", "inductor", "both"), default="both"
    )
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--accuracy-only", action="store_true")
    parser.add_argument("--load-only", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive; warmup nonnegative")
    if "all" in args.models:
        args.models = list(MODEL_CHOICES)
    # Preserve manifest order for deterministic scorecards.
    order = {name: i for i, name in enumerate(MODEL_CHOICES)}
    args.models = sorted(dict.fromkeys(args.models), key=order.__getitem__)
    return args


def tests_for_arg(test: str) -> list[str]:
    return ["eval", "train"] if test == "both" else [test]


def modes_for_arg(cpp_numerics: str) -> list[str]:
    return ["strict_eager", "inductor"] if cpp_numerics == "both" else [cpp_numerics]


# ---------------------------------------------------------------------------
# Genuine factory construction (no toy substitution).
# ---------------------------------------------------------------------------


def construct_benchmark(model_name: str, test: str, batch_size: int) -> Any:
    """Construct the genuine TorchBench BenchmarkModel for one case."""
    module = importlib.import_module(f"torchbenchmark.models.{model_name}")
    benchmark = module.Model(
        test=test, device=DEVICE, batch_size=batch_size, extra_args=list(EXTRA_ARGS)
    )
    if model_name == "hf_GPT2" and test == "train":
        _cache_native_loss_function(benchmark)
    return benchmark


def _cache_native_loss_function(benchmark: Any) -> None:
    """Cache the library-selected loss through its public property setter.

    Recent Transformers chooses the default loss inside a logging property,
    which breaks fullgraph compilation for both stock and CPP. Resolve that
    choice before compilation; never substitute a different loss callable.
    """
    module, _ = benchmark.get_module()
    prop = inspect.getattr_static(module, "loss_function", None)
    if not isinstance(prop, property) or prop.fset is None:
        return
    resolved = module.loss_function
    if not callable(resolved):
        raise TypeError("native Transformers loss function is not callable")
    module.loss_function = resolved
    if module.loss_function is not resolved:
        raise RuntimeError("native loss cache changed the selected callable")
    benchmark._loops_ir_preparation = (
        "cache identical native loss_function through its public setter",
    )


def inventory_benchmark(
    benchmark: Any, model_name: str, test: str, batch_size: int
) -> dict[str, Any]:
    import torch
    from torch.utils._pytree import tree_leaves

    try:
        module_path = inspect.getfile(type(benchmark))
    except Exception:
        module_path = f"torchbenchmark.models.{model_name}"
    try:
        module_cls_path = f"{type(benchmark).__module__}.{type(benchmark).__name__}"
    except Exception:
        module_cls_path = model_name
    try:
        mod, example_inputs = benchmark.get_module()
    except Exception as error:
        return {
            "model": model_name,
            "test": test,
            "batch_size": batch_size,
            "module_path": module_path,
            "module_class": module_cls_path,
            "construction_error": f"{type(error).__name__}: {error}",
        }
    try:
        n_params = sum(int(p.numel()) for p in mod.parameters())
    except Exception:
        n_params = None
    leaves: list[dict[str, Any]] = []
    try:
        for leaf in tree_leaves(example_inputs):
            if isinstance(leaf, torch.Tensor):
                leaves.append(
                    {
                        "shape": [int(s) for s in leaf.shape],
                        "strides": [int(s) for s in leaf.stride()],
                        "dtype": str(leaf.dtype),
                        "device": str(leaf.device),
                    }
                )
            else:
                leaves.append(
                    {"python_type": type(leaf).__name__, "repr": repr(leaf)[:200]}
                )
    except Exception as error:
        leaves = [{"error": f"{type(error).__name__}: {error}"}]
    return {
        "model": model_name,
        "test": test,
        "batch_size": batch_size,
        "batch_source": "suite-manifest-exact-source-default",
        "preparation": list(getattr(benchmark, "_loops_ir_preparation", ())),
        "extra_args": list(EXTRA_ARGS),
        "module_path": module_path,
        "module_class": module_cls_path,
        "num_parameters": n_params,
        "inputs": leaves,
    }


# ---------------------------------------------------------------------------
# State snapshots: deep clones only (never live state_dict aliases),
# including CPU/CUDA RNG, Python random, NumPy (when the genuine sources
# use it), original inputs/auxiliaries and the real optimizer/scheduler.
# ---------------------------------------------------------------------------


def _deep_clone(obj: Any) -> Any:
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: _deep_clone(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [_deep_clone(v) for v in obj]
        return type(obj)(seq) if isinstance(obj, tuple) else seq
    try:
        return copy.deepcopy(obj)
    except Exception:
        return obj


def _find_schedulers(benchmark: Any) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for attr in ("lr_scheduler", "optim_schedule", "scheduler"):
        try:
            obj = getattr(benchmark, attr, None)
        except Exception:
            obj = None
        if obj is not None:
            found[attr] = obj
    try:
        inner = getattr(benchmark, "model", None)
        for attr in ("optim_schedule", "lr_scheduler", "scheduler"):
            try:
                obj = getattr(inner, attr, None)
            except Exception:
                obj = None
            if obj is not None and attr not in found:
                found[f"model.{attr}"] = obj
    except Exception:
        pass
    return found


_AUX_ATTRS = ("example_inputs", "is_next", "bert_label", "targets", "example_outputs")


def snapshot_state(benchmark: Any) -> dict[str, Any]:
    """Snapshot model/optimizer/scheduler/inputs/RNG as typed fingerprints.

    Tensors freeze to dtype/shape/strides/bytes (layout is never erased);
    optimizer scalars (LR/momentum/group params) freeze via exact typed
    leaves; scheduler post-state is explicit (state_dict or every numeric
    attr plus inner-Adam state). Unsupported state fails loudly with an
    ``__error__`` marker (never a silent None); the comparator treats error
    markers as diffs and restore raises RestorationError.
    """
    import torch
    from torch.utils._pytree import tree_map

    snap: dict[str, Any] = {}
    try:
        mod, _ = benchmark.get_module()
    except Exception as error:
        snap["model_state_error"] = f"{type(error).__name__}: {error}"
        return snap
    target = _unwrap_for_state(mod)
    try:
        sd = target.state_dict()
        frozen_params: dict[str, Any] = {}
        for k, v in sd.items():
            if isinstance(v, torch.Tensor):
                frozen_params[k] = _tensor_fingerprint(v)
            else:
                try:
                    frozen_params[k] = _freeze_value(v)
                except Exception as error:
                    frozen_params[k] = {"__error__": f"{type(error).__name__}: {error}"}
        snap["model_state"] = frozen_params
    except Exception as error:
        snap["model_state_error"] = f"{type(error).__name__}: {error}"
    grads: dict[str, Any] = {}
    try:
        for name, param in target.named_parameters():
            g = param.grad
            if g is None:
                grads[name] = None
            elif isinstance(g, torch.Tensor):
                grads[name] = _tensor_fingerprint(g)
            else:
                grads[name] = {"__error__": f"non-tensor grad {type(g).__name__}"}
    except Exception as error:
        grads["__error__"] = f"{type(error).__name__}: {error}"
    snap["grads"] = grads
    try:
        buffers: dict[str, Any] = {}
        for name, buf in target.named_buffers():
            if isinstance(buf, torch.Tensor):
                buffers[name] = _tensor_fingerprint(buf)
            else:
                try:
                    buffers[name] = _freeze_value(buf)
                except Exception as error:
                    buffers[name] = {"__error__": f"{type(error).__name__}: {error}"}
        snap["buffers"] = buffers
    except Exception as error:
        snap["buffers"] = {"__error__": f"{type(error).__name__}: {error}"}
    try:
        opt = benchmark.get_optimizer()
    except Exception:
        opt = None
    snap["has_optimizer"] = opt is not None
    if opt is not None:
        try:
            snap["optimizer_state"] = _freeze_value(opt.state_dict())
        except Exception as error:
            snap["optimizer_state_error"] = f"{type(error).__name__}: {error}"
        # Structural optimizer map (finding 1): ordered per-group parameter
        # NAMES from the owning module (never cross-variant id() equality).
        # load_state_dict maps by group order/index; names + shapes prove the
        # target owns the same parameters in the same structure.
        try:
            id_to_name = {id(p): n for n, p in target.named_parameters()}
            id_to_shape = {
                id(p): ([int(s) for s in p.shape], str(p.dtype))
                for p in target.parameters()
            }
            group_maps = []
            for gi, group in enumerate(opt.param_groups):
                names = []
                for p in group["params"]:
                    names.append(id_to_name.get(id(p)))
                opts = {}
                for k, v in group.items():
                    if k == "params":
                        continue
                    try:
                        opts[k] = _freeze_value(v)
                    except Exception as error:
                        opts[k] = {"__error__": f"{type(error).__name__}: {error}"}
                group_maps.append({"group": gi, "param_names": names, "options": opts})
            snap["optimizer_param_map"] = group_maps
            snap["optimizer_param_shapes"] = {
                n: id_to_shape.get(i) for i, n in id_to_name.items()
            }
            snap["optimizer_class"] = type(opt).__name__
        except Exception as error:
            snap["optimizer_param_map_error"] = f"{type(error).__name__}: {error}"
    sched_states: dict[str, Any] = {}
    for name, sched in _find_schedulers(benchmark).items():
        sched_states[name] = _freeze_scheduler(sched)
    snap["scheduler_states"] = sched_states
    snap["schedulers"] = sched_states
    aux: dict[str, Any] = {}
    for attr in _AUX_ATTRS:
        for holder, prefix in (
            (benchmark, ""),
            (getattr(benchmark, "model", None), "model."),
        ):
            if holder is None:
                continue
            try:
                val = getattr(holder, attr, None)
            except Exception:
                continue
            if val is None:
                continue
            key = f"{prefix}{attr}"
            if key in aux:
                continue
            try:
                # Single freezing pass: tensors become typed fingerprints,
                # every other leaf freezes exactly. Already-frozen forms
                # (from a previous pass) pass through untouched -- never a
                # double-freeze that would wrap fingerprints in __dict__.
                def _freeze_leaf(x: Any) -> Any:
                    if isinstance(x, torch.Tensor):
                        return _tensor_fingerprint(x)
                    if isinstance(x, (FrozenMapping, FrozenKey)):
                        return x
                    if isinstance(x, dict) and (
                        x.get("__tensor__") is True
                        or "__seq__" in x
                        or "__float_bits__" in x
                        or "__error__" in x
                        or "__ndarray__" in x
                        or "__npscalar__" in x
                    ):
                        return x
                    return _freeze_value(x)

                aux[key] = tree_map(_freeze_leaf, val)
            except Exception:
                try:
                    aux[key] = _freeze_value(val)
                except Exception as error:
                    aux[key] = {"__error__": f"{type(error).__name__}: {error}"}
    snap["aux"] = aux
    try:
        snap["torch_rng"] = _tensor_fingerprint(torch.get_rng_state())
    except Exception as error:
        snap["torch_rng_error"] = f"{type(error).__name__}: {error}"
    try:
        if torch.cuda.is_available():
            snap["cuda_rng"] = [
                _tensor_fingerprint(s) for s in torch.cuda.get_rng_state_all()
            ]
        else:
            snap["cuda_rng"] = []
    except Exception as error:
        snap["cuda_rng_error"] = f"{type(error).__name__}: {error}"
    try:
        snap["python_rng"] = {
            "__python_rng__": {"repr": repr(random.getstate())},
            "state": _freeze_value(random.getstate()),
        }
    except Exception as error:
        snap["python_rng_error"] = f"{type(error).__name__}: {error}"
    try:
        import numpy as _np

        state = _np.random.get_state()
        snap["numpy_rng"] = {
            "__numpy_rng__": {"algorithm": str(state[0])},
            "state": _freeze_value(state),
        }
        snap["numpy_used"] = True
    except Exception:
        snap["numpy_used"] = False
    return snap


class RestorationError(RuntimeError):
    """Named-stage snapshot-restore failure; vetoes accuracy and timing."""


def _restore_scheduler_entry(
    benchmark: Any, name: str, entry: Any, *, param_ids: set[int] | None = None
) -> None:
    """Restore one scheduler fingerprint; raise RestorationError on any gap.

    Plain-attribute schedulers (e.g. BERT's ScheduledOptim) restore every
    numeric attr in place and reconcile the inner optimizer: by Parameter
    identity when the harness preserved refs, else in place into the current
    inner optimizer. Known slotted/aux-only attrs are reported, never silent
    bypass.
    """
    _ = param_ids
    if not isinstance(entry, dict) or "__error__" in entry:
        raise RestorationError(
            f"scheduler[{name}]: no restorable fingerprint "
            f"({entry if isinstance(entry, dict) else type(entry).__name__})"
        )
    holder = benchmark
    attr = name
    if "." in name:
        _, attr = name.split(".", 1)
        holder = getattr(benchmark, "model", benchmark)
    try:
        sched = getattr(holder, attr, None)
    except Exception as error:
        raise RestorationError(
            f"scheduler[{name}]: lookup failed ({type(error).__name__}: {error})"
        ) from error
    if sched is None:
        raise RestorationError(f"scheduler[{name}]: holder has no such scheduler")
    kind = entry.get("kind")
    if kind == "state_dict":
        if not hasattr(sched, "load_state_dict"):
            raise RestorationError(
                f"scheduler[{name}]: snapshot has state_dict but live "
                "scheduler lacks load_state_dict"
            )
        try:
            sched.load_state_dict(_thaw_value(entry.get("state")))
        except Exception as error:
            raise RestorationError(
                f"scheduler[{name}]: load_state_dict failed "
                f"({type(error).__name__}: {error})"
            ) from error
        return
    if kind == "plain_attrs":
        attrs = entry.get("attrs")
        if not isinstance(attrs, dict):
            raise RestorationError(f"scheduler[{name}]: plain_attrs missing attrs")
        for key in sorted(attrs, key=str):
            frozen = attrs[key]
            if isinstance(frozen, dict) and "__error__" in frozen:
                raise RestorationError(
                    f"scheduler[{name}].{key}: uncaptured attr ({frozen['__error__']})"
                )
            try:
                current = getattr(sched, key, "__missing__")
            except Exception as error:
                raise RestorationError(
                    f"scheduler[{name}].{key}: lookup failed "
                    f"({type(error).__name__}: {error})"
                ) from error
            if current == "__missing__":
                raise RestorationError(
                    f"scheduler[{name}].{key}: live scheduler lost the attribute"
                )
            try:
                setattr(sched, key, _thaw_value(frozen))
            except Exception as error:
                raise RestorationError(
                    f"scheduler[{name}].{key}: restore failed "
                    f"({type(error).__name__}: {error})"
                ) from error
        inner_frozen = entry.get("inner_optimizer")
        if inner_frozen is not None:
            if isinstance(inner_frozen, dict) and "__error__" in inner_frozen:
                raise RestorationError(
                    "scheduler[{name}]: inner optimizer uncaptured "
                    f"({inner_frozen['__error__']})".format(name=name)
                )
            inner = getattr(sched, "_optimizer", None)
            if inner is None or not hasattr(inner, "load_state_dict"):
                raise RestorationError(
                    f"scheduler[{name}]: live scheduler lost its inner optimizer"
                )
            try:
                inner.load_state_dict(_thaw_value(inner_frozen))
            except Exception as error:
                raise RestorationError(
                    f"scheduler[{name}]: inner optimizer restore failed "
                    f"({type(error).__name__}: {error})"
                ) from error
        return
    raise RestorationError(f"scheduler[{name}]: unknown kind {kind!r}")


def _thaw_value(frozen: Any) -> Any:
    """Thaw a frozen value back into live Python/torch objects (restore path)."""
    import struct

    if frozen is None or isinstance(frozen, (str, int)):
        return frozen
    if isinstance(frozen, bytes):
        return frozen
    if isinstance(frozen, FrozenMapping):
        return _thaw_mapping(frozen)
    if isinstance(frozen, FrozenKey):
        return _thaw_key(frozen)
    if isinstance(frozen, dict):
        if set(frozen.keys()) == {"__float_bits__"}:
            try:
                return struct.unpack(">d", bytes.fromhex(frozen["__float_bits__"]))[0]
            except Exception as error:
                raise RestorationError(
                    f"thaw float bits ({type(error).__name__}: {error})"
                ) from error
        if frozen.get("__tensor__") is True:
            return _reconstruct_tensor(frozen)
        if "__ndarray__" in frozen:
            try:
                import numpy as _np

                spec = frozen["__ndarray__"]
                # Writable copy: np.frombuffer is read-only and
                # np.random.set_state requires a writable uint32 array.
                arr = _np.frombuffer(
                    bytearray(bytes(spec["bytes"])), dtype=_np.dtype(spec["dtype"])
                ).reshape(tuple(int(s) for s in spec["shape"]))
                return _np.array(arr, copy=True)
            except Exception as error:
                raise RestorationError(
                    f"thaw ndarray ({type(error).__name__}: {error})"
                ) from error
        if "__npscalar__" in frozen:
            try:
                import numpy as _np

                spec = frozen["__npscalar__"]
                return _np.frombuffer(
                    bytearray(bytes(spec["bytes"])), dtype=_np.dtype(spec["dtype"])
                )[()]
            except Exception as error:
                raise RestorationError(
                    f"thaw numpy scalar ({type(error).__name__}: {error})"
                ) from error
        if "__seq__" in frozen:
            seq = [_thaw_value(v) for v in frozen["__seq__"]]
            return tuple(seq) if frozen.get("__tuple__") else seq
        if "__dict__" in frozen:
            try:
                return {k: _thaw_value(v) for k, v in frozen["__dict__"].items()}
            except Exception as error:
                raise RestorationError(
                    f"thaw dict ({type(error).__name__}: {error})"
                ) from error
        if "__error__" in frozen:
            raise RestorationError(f"frozen error marker: {frozen['__error__']}")
        try:
            return {k: _thaw_value(v) for k, v in frozen.items()}
        except Exception as error:
            raise RestorationError(
                f"thaw mapping ({type(error).__name__}: {error})"
            ) from error
    if isinstance(frozen, (list, tuple)):
        seq = [_thaw_value(v) for v in frozen]
        return tuple(seq) if isinstance(frozen, tuple) else seq
    raise RestorationError(
        f"cannot thaw value of type {type(frozen).__name__}; no hidden defaults"
    )


def _thaw_python_state(frozen: Any) -> Any:
    """Thaw a frozen Python random state (tuple data preserved exactly).

    Python's getstate() is (version:int, state:tuple[int,...], gauss:None|float);
    no bytes form is needed -- key equality over the thawed ints is exact.
    """
    if isinstance(frozen, dict) and "state" in frozen:
        inner = frozen["state"]
        if isinstance(inner, dict) and "__seq__" in inner:
            return tuple(_thaw_value(v) for v in inner["__seq__"])
        return _thaw_value(inner)
    if isinstance(frozen, tuple):
        return frozen
    return _thaw_value(frozen)


def _thaw_numpy_state(frozen: Any) -> Any:
    """Thaw a frozen NumPy RandomState tuple (uint32 array via typed bytes)."""
    if isinstance(frozen, dict) and "state" in frozen:
        return _thaw_value(frozen["state"])
    return _thaw_value(frozen)


def _continuation_draws() -> dict[str, Any]:
    """Draw continuation values from every RNG to capture divergence.

    Called after a training step on both variants from the same restored
    state: identical draws prove the RNG streams were not diverged by the
    step; differing draws name the diverged stream. Strict compares the raw
    continuation values (torch/CUDA uint8 bytes, Python ints, NumPy uint32
    bytes); normal compares quantitatively where applicable.
    """
    import torch

    draws: dict[str, Any] = {}
    try:
        draws["torch"] = _tensor_fingerprint(torch.randint(0, 2**31, (16,)))
    except Exception as error:
        draws["torch"] = {"__error__": f"{type(error).__name__}: {error}"}
    try:
        if torch.cuda.is_available():
            draws["cuda"] = _tensor_fingerprint(
                torch.randint(0, 2**31, (16,), device="cuda")
            )
        else:
            draws["cuda"] = []
    except Exception as error:
        draws["cuda"] = {"__error__": f"{type(error).__name__}: {error}"}
    try:
        draws["python"] = _freeze_value((random.random(), random.getrandbits(64)))
    except Exception as error:
        draws["python"] = {"__error__": f"{type(error).__name__}: {error}"}
    try:
        import numpy as _np

        draws["numpy"] = _freeze_value((_np.random.randint(0, 2**31, size=8),))
    except Exception:
        draws["numpy"] = {"__error__": "numpy unavailable"}
    return draws


def restore_state(benchmark: Any, snap: dict[str, Any]) -> None:
    """Restore a snapshot with named-stage errors (fail closed).

    Every needed component is restored exactly: model (strict load),
    grads, optimizer (with Parameter-identity check), schedulers, aux
    tensors, and all RNG states. Any failure raises :class:`RestorationError`
    naming the stage; the caller must veto accuracy/timing and record the
    error, never silently proceed with a half-restored variant.
    """
    import torch
    from torch.utils._pytree import tree_map

    if "model_state_error" in snap:
        raise RestorationError(f"model: snapshot missing ({snap['model_state_error']})")
    if "model_state" not in snap:
        raise RestorationError("model: snapshot has no model_state")
    mod, _ = benchmark.get_module()
    target = _unwrap_for_state(mod)
    try:
        frozen_state = snap["model_state"]
        live_state = target.state_dict()
        if set(frozen_state.keys()) != set(live_state.keys()):
            raise RestorationError(
                "model: key mismatch "
                f"(missing={sorted(set(live_state) - set(frozen_state))[:5]}, "
                f"extra={sorted(set(frozen_state) - set(live_state))[:5]})"
            )
        restored = {k: _thaw_value(v) for k, v in frozen_state.items()}
        target.load_state_dict(restored, strict=True)
    except RestorationError:
        raise
    except Exception as error:
        raise RestorationError(
            f"model: strict load failed ({type(error).__name__}: {error})"
        ) from error
    try:
        named = dict(target.named_parameters())
        for name, frozen_grad in snap.get("grads", {}).items():
            if name == "__error__":
                raise RestorationError(
                    f"grads: snapshot capture failed ({frozen_grad})"
                )
            if name not in named:
                raise RestorationError(f"grads: unknown parameter {name}")
            if frozen_grad is None:
                named[name].grad = None
            else:
                named[name].grad = _thaw_value(frozen_grad)
                if not isinstance(named[name].grad, torch.Tensor):
                    raise RestorationError(f"grads: {name} thawed to non-tensor")
    except RestorationError:
        raise
    except Exception as error:
        raise RestorationError(
            f"grads: restore failed ({type(error).__name__}: {error})"
        ) from error
    if snap.get("has_optimizer"):
        try:
            opt = benchmark.get_optimizer()
        except Exception as error:
            raise RestorationError(
                f"optimizer: lookup failed ({type(error).__name__}: {error})"
            ) from error
        if opt is None:
            raise RestorationError("optimizer: snapshot has one, live has none")
        # Structural check (finding 1): same optimizer class, same group
        # count/lengths, every target optimizer param owned by the live
        # target module (same-object), source name map consistent with the
        # target's names, per-name shapes equal (catches swapped group
        # order that group-index load_state_dict would mis-apply).
        # Cross-variant id() equality is NEVER required: different genuine
        # factories own different Parameter objects by design.
        if "optimizer_class" in snap and snap["optimizer_class"] != type(opt).__name__:
            raise RestorationError(
                "optimizer: class changed "
                f"({snap['optimizer_class']} -> {type(opt).__name__})"
            )
        live_named = dict(target.named_parameters())
        live_ids = {id(p) for p in target.parameters()}
        target_groups = opt.param_groups
        source_map = snap.get("optimizer_param_map")
        if source_map is not None:
            if len(source_map) != len(target_groups):
                raise RestorationError(
                    "optimizer: group count changed "
                    f"({len(source_map)} -> {len(target_groups)})"
                )
            source_shapes = snap.get("optimizer_param_shapes") or {}
            for sgroup, tgroup in zip(source_map, target_groups):
                snames = sgroup.get("param_names") or []
                tparams = list(tgroup.get("params") or [])
                if len(snames) != len(tparams):
                    raise RestorationError(
                        f"optimizer: group {sgroup.get('group')} length changed "
                        f"({len(snames)} -> {len(tparams)})"
                    )
                live_names = []
                for p in tparams:
                    if id(p) not in live_ids:
                        raise RestorationError(
                            "optimizer: target optimizer owns a parameter "
                            "outside the live target module"
                        )
                    lname = next((n for n, lp in live_named.items() if lp is p), None)
                    if lname is None:
                        raise RestorationError(
                            "optimizer: target param not found by identity"
                        )
                    live_names.append(lname)
                for sname, lname in zip(snames, live_names):
                    if sname is None:
                        raise RestorationError(
                            "optimizer: source param had no owning-module name"
                        )
                    if sname != lname:
                        raise RestorationError(
                            "optimizer: param-name mapping inconsistent "
                            f"(source {sname} -> target {lname}); possible "
                            "swapped group order"
                        )
                    sshape = source_shapes.get(sname)
                    live_p = live_named[lname]
                    tshape = ([int(s) for s in live_p.shape], str(live_p.dtype))
                    if sshape is not None and list(sshape) != list(tshape):
                        raise RestorationError(
                            f"optimizer: param {lname} shape/dtype changed "
                            f"({sshape} -> {tshape})"
                        )
        # Local same-object identity: target optimizer params must be the
        # live module's objects before AND after restore (values copy in
        # place; load_state_dict never rebinds).
        before_objs = sorted(id(p) for g in target_groups for p in g["params"])
        if "optimizer_state" not in snap:
            raise RestorationError("optimizer: snapshot missing optimizer_state")
        try:
            opt.load_state_dict(_thaw_value(snap["optimizer_state"]))
        except RestorationError:
            raise
        except Exception as error:
            raise RestorationError(
                f"optimizer: load_state_dict failed ({type(error).__name__}: {error})"
            ) from error
        after_objs = sorted(id(p) for g in opt.param_groups for p in g["params"])
        if before_objs != after_objs:
            raise RestorationError(
                "optimizer: load_state_dict rebound Parameter objects "
                "(identity must be preserved)"
            )
        for g in opt.param_groups:
            for p in g["params"]:
                if id(p) not in live_ids:
                    raise RestorationError(
                        "optimizer: post-restore param outside live module"
                    )
    for name, entry in snap.get("scheduler_states", {}).items():
        _restore_scheduler_entry(benchmark, name, entry)
    for key, frozen in snap.get("aux", {}).items():
        if isinstance(frozen, dict) and "__error__" in frozen:
            raise RestorationError(f"aux[{key}]: uncaptured ({frozen['__error__']})")
        try:
            if "." in key:
                _, attr = key.split(".", 1)
                holder = getattr(benchmark, "model", benchmark)
            else:
                attr = key
                holder = benchmark
            try:
                restored = tree_map(
                    lambda x: x.detach().clone() if isinstance(x, torch.Tensor) else x,
                    _thaw_value(frozen),
                )
            except Exception:
                restored = _thaw_value(frozen)
            setattr(holder, attr, restored)
        except RestorationError:
            raise
        except Exception as error:
            raise RestorationError(
                f"aux[{key}]: restore failed ({type(error).__name__}: {error})"
            ) from error
    if "torch_rng" in snap:
        try:
            torch.set_rng_state(_thaw_value(snap["torch_rng"]))
        except RestorationError:
            raise
        except Exception as error:
            raise RestorationError(
                f"rng[torch]: restore failed ({type(error).__name__}: {error})"
            ) from error
    else:
        raise RestorationError("rng[torch]: snapshot missing torch_rng")
    if "torch_rng_error" in snap:
        raise RestorationError(
            f"rng[torch]: snapshot error ({snap['torch_rng_error']})"
        )
    if torch.cuda.is_available():
        if "cuda_rng" not in snap:
            raise RestorationError("rng[cuda]: snapshot missing cuda_rng")
        try:
            torch.cuda.set_rng_state_all(
                [
                    s if isinstance(s, torch.Tensor) else _reconstruct_tensor(s)
                    for s in snap["cuda_rng"]
                ]
            )
        except RestorationError:
            raise
        except Exception as error:
            raise RestorationError(
                f"rng[cuda]: restore failed (fail closed; "
                f"{type(error).__name__}: {error})"
            ) from error
    if "cuda_rng_error" in snap:
        raise RestorationError(f"rng[cuda]: snapshot error ({snap['cuda_rng_error']})")
    if "python_rng" in snap:
        try:
            random.setstate(_thaw_python_state(snap["python_rng"]))
        except RestorationError:
            raise
        except Exception as error:
            raise RestorationError(
                f"rng[python]: restore failed ({type(error).__name__}: {error})"
            ) from error
    else:
        raise RestorationError("rng[python]: snapshot missing python_rng")
    if snap.get("numpy_used"):
        if "numpy_rng" not in snap:
            raise RestorationError("rng[numpy]: snapshot missing numpy_rng")
        try:
            import numpy as _np

            _np.random.set_state(_thaw_numpy_state(snap["numpy_rng"]))
        except RestorationError:
            raise
        except Exception as error:
            raise RestorationError(
                f"rng[numpy]: restore failed ({type(error).__name__}: {error})"
            ) from error


def _is_staged(benchmark: Any) -> bool:
    return (
        hasattr(benchmark, "forward")
        and hasattr(benchmark, "backward")
        and hasattr(benchmark, "optimizer_step")
        and getattr(benchmark, "train", None) is None
    )


class _LossObserver:
    """Temporary stage/observer hook using actual native loss semantics.

    Wraps the model's real loss callable (``loss_fn`` / trainer
    ``criterion`` / Timm ``cfg.loss``) to record the genuine loss tensor
    observed during ``invoke()``/``train()``. Removed before timing; if no
    loss callable exists the capability is reported as unavailable and no
    loss is fabricated.
    """

    def __init__(self, benchmark: Any):
        self.benchmark = benchmark
        self.targets: list[tuple[Any, str]] = []
        self.originals: list[tuple[Any, str, Any]] = []
        self.last_loss: Any = None
        self.via: str = "unavailable"

    def __enter__(self):
        bm = self.benchmark
        candidates: list[tuple[Any, str]] = [(bm, "loss_fn")]
        try:
            inner = bm.model
        except Exception:
            inner = None
        if inner is not None:
            candidates.append((inner, "criterion"))
        try:
            cfg = bm.cfg
        except Exception:
            cfg = None
        try:
            has_loss = callable(getattr(cfg, "loss", None))
        except Exception:
            has_loss = False
        if cfg is not None and has_loss:
            candidates.append((cfg, "loss"))
        for holder, attr in candidates:
            if holder is None:
                continue
            try:
                original = getattr(holder, attr, None)
            except Exception:
                continue
            if not callable(original):
                continue
            if getattr(original, "_loops_ir_suite_observed", False):
                continue

            def _wrapped(
                *args: Any, _orig: Any = original, _self: Any = self, **kwargs: Any
            ) -> Any:
                import torch as _torch

                out = _orig(*args, **kwargs)
                try:
                    _self.last_loss = (
                        out.detach().clone() if isinstance(out, _torch.Tensor) else None
                    )
                except Exception:
                    _self.last_loss = None
                return out

            _wrapped._loops_ir_suite_observed = True  # type: ignore[attr-defined]
            try:
                setattr(holder, attr, _wrapped)
                self.originals.append((holder, attr, original))
                if self.via == "unavailable":
                    holder_name = type(holder).__name__
                    self.via = f"{holder_name}.{attr}"
            except Exception:
                continue
        return self

    def __exit__(self, *exc: Any) -> None:
        for holder, attr, original in self.originals:
            try:
                setattr(holder, attr, original)
            except Exception:
                continue
        self.originals.clear()


# ---------------------------------------------------------------------------
# Compiler wrapper: get_module + set_module preserving Parameter identity
# and optimizer refs; AOT compiles backward; optimizer may stay eager.
# ---------------------------------------------------------------------------


def _param_id_set(module: Any) -> set[int]:
    try:
        return {id(p) for p in module.parameters()}
    except Exception:
        return set()


def _summarize_seam_calls(seam_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Honest per-graph evidence summary from seam-call records.

    Shared by the success and failure paths so a rejected compile still
    records which graphs ran, their admitted kinds, and their errors
    (rejection evidence is data, never a pass).
    """

    def _kind(call: dict[str, Any]) -> str:
        return str(call.get("graph_kind") or "")

    backward_evidence = any(
        call.get("ok") is True
        and _kind(call) == "backward"
        and call.get("graph_kwargs", {}).get("is_backward") is True
        for call in seam_calls
    )
    forward_evidence = any(
        call.get("ok") is True and _kind(call) == "forward" for call in seam_calls
    )
    inference_evidence = any(
        call.get("ok") is True and _kind(call) == "inference" for call in seam_calls
    )
    if not backward_evidence:
        backward_evidence = any(
            call.get("ok") is True
            and not _kind(call)
            and call.get("graph_kwargs", {}).get("is_backward") is True
            for call in seam_calls
        )
    successful_calls = [c for c in seam_calls if c.get("ok") is True]
    kinds = sorted({_kind(c) for c in successful_calls})

    def _count(call: dict[str, Any], name: str) -> int:
        value = call.get(name)
        return value if type(value) is int and value >= 0 else 0

    generated = sum(_count(c, "num_generated_kernels") for c in successful_calls)
    before_complete = bool(successful_calls) and all(
        type(c.get("kernels_before_fusion")) is int
        and c["kernels_before_fusion"] >= _count(c, "num_generated_kernels")
        for c in successful_calls
    )
    before_fusion = (
        sum(_count(c, "kernels_before_fusion") for c in successful_calls)
        if before_complete
        else None
    )
    coverage = {
        "scope": "all_successful_seam_graphs",
        "num_generated_kernels": generated,
        "kernels_before_fusion": before_fusion,
        "kernels_removed_by_fusion": before_fusion - generated
        if before_fusion is not None
        else None,
        "num_extern_calls": sum(
            _count(c, "num_extern_calls") for c in successful_calls
        ),
        "function_names": [
            name for call in successful_calls for name in call.get("function_names", [])
        ],
    }
    try:
        trace_complete = bool(seam_calls) and all(
            isinstance(c.get("pipeline_trace_stages"), int)
            and c["pipeline_trace_stages"] >= 1
            for c in seam_calls
            if c.get("ok") is True
        )
    except Exception:
        trace_complete = False
    return {
        "num_seam_graphs": len(seam_calls),
        "num_seam_graphs_ok": sum(1 for c in seam_calls if c.get("ok") is True),
        "forward_cpp_evidence": bool(forward_evidence),
        "backward_cpp_evidence": bool(backward_evidence),
        "inference_cpp_evidence": bool(inference_evidence),
        "backend_graph_kind": "+".join(k for k in kinds if k) or "unknown",
        "formal_trace_complete": bool(trace_complete),
        "collector_retains_only_last_graph": True,
        "coverage": coverage,
        "calls": list(seam_calls),
    }


def compile_variant(benchmark: Any, mode: str, test: str) -> dict[str, Any]:
    """Compile the module from ``get_module()`` and reinstall via ``set_module()``.

    Returns a coverage record. Raises on failure (caller records per-case
    errors; training cases additionally require backward-CPP evidence to
    pass, enforced at verdict time, not here).
    """
    import torch
    from torch._inductor import config as inductor_config
    from torch._inductor.loops_ir_backend import (
        _Collector,
        _current_collector,
        LoopsIrArtifacts,
    )

    info: dict[str, Any] = {
        "mode": mode,
        "test": test,
        "optimizer_compiled": False,
        "optimizer_eager": True,
    }
    mod, _ = benchmark.get_module()
    try:
        opt_before = benchmark.get_optimizer()
    except Exception:
        opt_before = None
    try:
        opt_param_ids_before = (
            sorted(id(p) for g in opt_before.param_groups for p in g["params"])
            if opt_before is not None
            else []
        )
    except Exception:
        opt_param_ids_before = []
    info["optimizer_present"] = opt_before is not None
    info["param_ids_before"] = sorted(_param_id_set(mod))

    options: dict[str, Any] = {"experimental_cpp_ir": True, "triton.cudagraphs": False}
    if mode:
        options["experimental_cpp_ir_numerics"] = mode
    info["options"] = dict(options)

    import torch._inductor.loops_ir_backend as _backend_mod

    orig_seam = _backend_mod.compile_fx_loops_ir_output
    seam_calls: list[dict[str, Any]] = []

    def _counting_seam(gm: Any, example_inputs: Any, **kwargs: Any) -> Any:
        entry: dict[str, Any] = {"ok": False}
        try:
            entry["num_nodes"] = sum(1 for _ in gm.graph.nodes)
            try:
                entry["graph_kwargs"] = {
                    k: (bool(v) if isinstance(v, bool) else str(v)[:120])
                    for k, v in (kwargs.get("graph_kwargs") or {}).items()
                }
            except Exception:
                entry["graph_kwargs"] = {}
        except Exception:
            pass
        seam_calls.append(entry)
        try:
            out = orig_seam(gm, example_inputs, **kwargs)
        except Exception as error:
            entry.update(
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error)[:2000],
                }
            )
            raise
        entry["ok"] = True
        try:
            artifacts = (
                _current_collector.get().artifacts if _current_collector.get() else None
            )
            if artifacts is not None:
                entry["num_generated_kernels"] = int(artifacts.num_generated_kernels)
                entry["kernels_before_fusion"] = int(artifacts.kernels_before_fusion)
                entry["num_extern_calls"] = int(artifacts.num_extern_calls)
                entry["function_names"] = list(artifacts.function_names)
                # Typed training admission (worker C, landed): real graph
                # kind + mandatory formal-pipeline trace per graph.
                entry["graph_kind"] = str(getattr(artifacts, "graph_kind", "inference"))
                try:
                    trace = getattr(artifacts, "pipeline_trace", ()) or ()
                    entry["pipeline_trace_stages"] = len(trace)
                    entry["pipeline_trace"] = [
                        {
                            "name": str(getattr(t, "name", "")),
                            "version": str(getattr(t, "version", "")),
                            "input_ir": str(getattr(t, "input_ir", "")),
                            "output_ir": str(getattr(t, "output_ir", "")),
                            "numerics": str(getattr(t, "numerics", "")),
                            "input_fingerprint": str(
                                getattr(t, "input_fingerprint", "")
                            ),
                            "output_fingerprint": str(
                                getattr(t, "output_fingerprint", "")
                            ),
                        }
                        for t in trace
                    ]
                except Exception:
                    entry["pipeline_trace_stages"] = None
                    entry["pipeline_trace"] = []
        except Exception:
            pass
        return out

    artifacts = LoopsIrArtifacts()
    _backend_mod.compile_fx_loops_ir_output = _counting_seam  # type: ignore[assignment]
    token = None
    compile_seconds: float | None = None
    # Any failure below (including a rejection raised inside the seam and
    # re-raised through torch.compile) carries the partial per-graph record
    # so the caller preserves rejection evidence (never a pass).
    try:
        compiled = torch.compile(
            mod, backend="inductor", fullgraph=True, dynamic=False, options=options
        )
        token = _current_collector.set(
            _Collector(artifacts, dumps=True, source=True, ir=True, statistics=True)
        )
        start = time.perf_counter()
        # Fresh compilation for honest artifact capture (never a cache hit
        # masquerading as coverage).
        with inductor_config.patch(
            {"fx_graph_cache": False, "fx_graph_remote_cache": False}
        ):
            if test == "eval":
                with torch.inference_mode():
                    _, _example = benchmark.get_module()
                    if isinstance(_example, dict):
                        compiled(**_example)
                    else:
                        compiled(*_flat_inputs_for_compile(benchmark))
            else:
                # Compile-time first run uses the genuine full step so AOT
                # sees forward+backward; state is restored by the caller.
                benchmark.set_module(compiled)
                try:
                    benchmark.invoke()
                finally:
                    pass
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        compile_seconds = time.perf_counter() - start
        # For eval the module was not yet installed; install now.
        if test == "eval":
            benchmark.set_module(compiled)
        else:
            # Already installed above; ensure identity (no re-wrap).
            pass
    except Exception as _compile_error:
        # Attach the partial per-graph record (rejection evidence included)
        # so the caller preserves it; the error itself still propagates.
        try:
            _compile_error._loops_ir_seam_calls = list(seam_calls)  # type: ignore[attr-defined]
        except Exception:
            pass
        raise
    finally:
        try:
            _backend_mod.compile_fx_loops_ir_output = orig_seam  # type: ignore[assignment]
        except Exception:
            pass
        if token is not None:
            try:
                _current_collector.reset(token)
            except Exception:
                pass
    info["compile_and_first_run_seconds"] = compile_seconds
    info["seam_calls"] = seam_calls
    # Per-graph record: the existing collector retains only the last graph;
    # per-call entries above are the honest per-graph hook record. Shared
    # helper so the failure path records the same evidence shape.
    summary = _summarize_seam_calls(seam_calls)
    info["num_seam_graphs"] = summary["num_seam_graphs"]
    info["coverage"] = summary["coverage"]
    info["triton_sources_available"] = sorted(artifacts.triton_sources)
    info["triton_ptx_available"] = sorted(artifacts.triton_ptx)
    # Last-graph formal provenance (the existing collector retains only the
    # last graph; per-call entries above are the per-graph record).
    info["last_graph_kind"] = str(getattr(artifacts, "graph_kind", "inference"))
    try:
        last_trace = getattr(artifacts, "pipeline_trace", ()) or ()
        info["last_pipeline_trace_stages"] = len(last_trace)
        info["last_pipeline_trace"] = [
            {
                "name": str(getattr(t, "name", "")),
                "version": str(getattr(t, "version", "")),
                "input_ir": str(getattr(t, "input_ir", "")),
                "output_ir": str(getattr(t, "output_ir", "")),
                "numerics": str(getattr(t, "numerics", "")),
            }
            for t in last_trace
        ]
    except Exception:
        info["last_pipeline_trace_stages"] = None
        info["last_pipeline_trace"] = []
    # Last-graph artifact texts (the existing collector retains only the
    # last graph). Returned for the caller to persist to disk; the caller
    # must NOT embed these blobs in the JSON scorecard (paths only).
    info["artifact_texts"] = {
        "program_dump": artifacts.program_dump,
        "optimized_program_dump": artifacts.optimized_program_dump,
        "kernel_dump": artifacts.kernel_dump,
        "tiled_dump": artifacts.tiled_dump,
        "triton_sources": dict(artifacts.triton_sources),
        "triton_ptx": dict(artifacts.triton_ptx),
    }

    # Backward/forward evidence from the typed training admission
    # (worker C, landed): shared helper (same shape as the failure path).
    summary = _summarize_seam_calls(seam_calls)
    info["backward_cpp_evidence"] = summary["backward_cpp_evidence"]
    info["forward_cpp_evidence"] = summary["forward_cpp_evidence"]
    info["inference_cpp_evidence"] = summary["inference_cpp_evidence"]
    info["backend_graph_kind"] = summary["backend_graph_kind"]
    info["num_seam_graphs"] = summary["num_seam_graphs"]
    info["num_seam_graphs_ok"] = summary["num_seam_graphs_ok"]
    info["formal_trace_complete"] = summary["formal_trace_complete"]
    info["collector_retains_only_last_graph"] = True
    info["pipeline_trace_stages"] = max(
        (
            c.get("pipeline_trace_stages")
            for c in seam_calls
            if isinstance(c.get("pipeline_trace_stages"), int)
        ),
        default=None,
    )
    # Parameter identity + optimizer refs preserved?
    try:
        mod_after, _ = benchmark.get_module()
        info["param_ids_after"] = sorted(_param_id_set(mod_after))
    except Exception:
        info["param_ids_after"] = []
    try:
        opt_after = benchmark.get_optimizer()
        opt_param_ids_after = (
            sorted(id(p) for g in opt_after.param_groups for p in g["params"])
            if opt_after is not None
            else []
        )
    except Exception:
        opt_after = None
        opt_param_ids_after = []
    info["param_identity_preserved"] = (
        bool(info["param_ids_before"])
        and info["param_ids_before"] == info["param_ids_after"]
    )
    info["optimizer_refs_preserved"] = (opt_before is None and opt_after is None) or (
        opt_before is not None
        and opt_after is not None
        and opt_param_ids_before == opt_param_ids_after
    )
    if info.get("coverage", {}).get("num_generated_kernels", 0) <= 0:
        error = RuntimeError(
            "no generated C++ IR kernels in successful graphs (coverage gate)"
        )
        # Preserve per-graph evidence on the exception so the caller records
        # rejection/attempt data even for failed compiles (never a pass).
        error._loops_ir_seam_calls = list(seam_calls)  # type: ignore[attr-defined]
        raise error
    return info


def _flat_inputs_for_compile(benchmark: Any) -> list[Any]:
    """Positional inputs for the eval compile-time first run.

    Uses the genuine example inputs. For dict inputs, fall back to the
    genuine driver path instead of guessing: raise so the caller uses
    benchmark.invoke().
    """
    _, example_inputs = benchmark.get_module()
    if isinstance(example_inputs, dict):
        raise RuntimeError("dict inputs use the genuine driver path (invoke)")
    if isinstance(example_inputs, (list, tuple)):
        return list(example_inputs)
    return [example_inputs]


def compile_stock_variant(benchmark: Any, test: str) -> dict[str, Any]:
    """Compile with stock Inductor (reference, same fullgraph/dynamic)."""
    import torch

    mod, _ = benchmark.get_module()
    options: dict[str, Any] = {"triton.cudagraphs": False}
    start = time.perf_counter()
    compiled = torch.compile(
        mod, backend="inductor", fullgraph=True, dynamic=False, options=options
    )
    if test == "eval":
        _, example_inputs = benchmark.get_module()
        with torch.inference_mode():
            if isinstance(example_inputs, dict):
                compiled(**example_inputs)
            else:
                args = (
                    list(example_inputs)
                    if isinstance(example_inputs, (list, tuple))
                    else [example_inputs]
                )
                compiled(*args)
    else:
        benchmark.set_module(compiled)
        benchmark.invoke()
        return {
            "options": options,
            "compile_and_first_run_seconds": time.perf_counter() - start,
            "installed": True,
        }
    benchmark.set_module(compiled)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {
        "options": options,
        "compile_and_first_run_seconds": time.perf_counter() - start,
        "installed": True,
    }


# ---------------------------------------------------------------------------
# Accuracy: eval outputs; train full post-state (strict bytewise + normal
# tolerances recorded separately; unrelated stochastic masks never compared
# with assert_close).
# ---------------------------------------------------------------------------


def _clone_output(obj: Any) -> Any:
    import torch
    from torch.utils._pytree import tree_map

    try:
        return tree_map(
            lambda x: x.detach().clone() if isinstance(x, torch.Tensor) else x, obj
        )
    except Exception:
        return None


def _stock_tolerance_and_flags(
    model_name: str, test: str, precision: str = "fp32"
) -> tuple[float, bool, dict[str, Any]]:
    """Existing model policy from get_tolerance_and_cosine_flag (no overrides).

    Uses ONLY the established stock API: env_check.get_tolerance_and_cosine_flag
    (authoritative source; only evalFP32 tol=1e-4 with both atol and rtol and
    trainCUDA tol=1e-3 with existing model metadata are exercised here; nothing
    is bumped or handwritten). Returns (tol, cos_similarity, record) where the
    record names the exact API for the report.
    """
    from torchbenchmark.util.env_check import (  # type: ignore[import]
        get_tolerance_and_cosine_flag,
    )

    import types as _types

    is_training = test == "train"
    current_device = DEVICE
    probe = _types.SimpleNamespace(
        dargs=_types.SimpleNamespace(precision=precision, use_cosine_similarity=False)
    )
    tolerance, cosine = get_tolerance_and_cosine_flag(
        probe, is_training, current_device, model_name
    )
    record = {
        "source": STOCK_TOLICY_SOURCE,
        "model": model_name,
        "test": test,
        "device": current_device,
        "precision": precision,
        "tol": float(tolerance),
        "cosine": bool(cosine),
    }
    return float(tolerance), bool(cosine), record


def _stock_same(
    correct_result: Any,
    new_result: Any,
    fp64_outputs: Any,
    *,
    tol: float,
    cos_similarity: bool,
    equal_nan: bool = False,
) -> tuple[bool, str]:
    """Established stock-effective comparator: torch._dynamo.utils.same.

    Policy (allclose first, then the FP64 RMSE 2x reference-error envelope,
    cosine only when the stock policy sets it) is the existing implementation
    -- never reimplemented here. Returns (pass, detail).
    """
    from torch._dynamo.utils import same as _stock_same_fn

    import io as _io
    import logging as _logging

    records: list[str] = []

    def _capture(msg: Any, *args: Any) -> None:
        try:
            records.append(str(msg) % tuple(args) if args else str(msg))
        except Exception:
            try:
                records.append(str(msg))
            except Exception:
                pass

    try:
        ok = bool(
            _stock_same_fn(
                correct_result,
                new_result,
                fp64_outputs,
                cos_similarity=cos_similarity,
                tol=tol,
                equal_nan=equal_nan,
                log_error=_capture,
            )
        )
    except Exception as error:
        return False, f"{type(error).__name__}: {str(error)[:500]}"
    _ = _io, _logging
    return ok, "; ".join(records[-3:])[:1000]


def _save_fp32_precision() -> tuple[Any, Any]:
    """Save raw fp32-precision backends (lossless).

    ``torch.get_float32_matmul_precision()`` returns a lossy string:
    ``set(get())`` flips cuda/mkldnn backends ('none','none')->('ieee','ieee').
    Save the raw backend tuples so restore is exact (no global leak).
    """
    import torch as _torch

    cuda_val: Any = None
    mkldnn_val: Any = None
    try:
        cuda_val = _torch.backends.cuda.matmul.fp32_precision
    except Exception:
        pass
    try:
        mkldnn_val = _torch.backends.mkldnn.matmul.fp32_precision
    except Exception:
        pass
    return cuda_val, mkldnn_val


def _restore_fp32_precision(saved: tuple[Any, Any]) -> None:
    """Restore raw fp32-precision backends exactly (never the lossy string)."""
    import torch as _torch

    cuda_val, mkldnn_val = saved
    try:
        if cuda_val is not None:
            _torch.backends.cuda.matmul.fp32_precision = cuda_val
    except Exception:
        pass
    try:
        if mkldnn_val is not None:
            _torch.backends.mkldnn.matmul.fp32_precision = mkldnn_val
    except Exception:
        pass


def _restore_global_settings(saved: dict[str, Any]) -> None:
    """Restore torch/CUDA global numerics recorded by numerical_settings().

    Never touches ``float32_matmul_precision`` here: the string form is a
    lossy round-trip (set(get()) flips cuda/mkldnn 'none'->'ieee'). Callers
    save/restore the raw backends via _save/_restore_fp32_precision.
    """
    import torch as _torch

    from torch._inductor import config as _cfg

    try:
        if "deterministic_algorithms" in saved:
            _torch.use_deterministic_algorithms(bool(saved["deterministic_algorithms"]))
    except Exception:
        pass
    try:
        if "cudnn_deterministic" in saved:
            _torch.backends.cudnn.deterministic = bool(saved["cudnn_deterministic"])
    except Exception:
        pass
    try:
        if "cudnn_benchmark" in saved:
            _torch.backends.cudnn.benchmark = bool(saved["cudnn_benchmark"])
    except Exception:
        pass
    try:
        if "cudnn_allow_tf32" in saved:
            _torch.backends.cudnn.allow_tf32 = bool(saved["cudnn_allow_tf32"])
    except Exception:
        pass
    try:
        if "matmul_allow_tf32" in saved:
            _torch.backends.cuda.matmul.allow_tf32 = bool(saved["matmul_allow_tf32"])
    except Exception:
        pass
    # NOTE: float32_matmul_precision intentionally NOT restored from the
    # lossy string -- see _save/_restore_fp32_precision.
    try:
        if saved.get("emulate_precision_casts") is not None:
            _cfg.emulate_precision_casts = bool(saved["emulate_precision_casts"])
    except Exception:
        pass


def _snapshot_rng_state() -> dict[str, Any]:
    """Save torch/CUDA/Python/NumPy RNG states (FP64-reference save side)."""
    import random as _random

    import torch as _torch

    saved: dict[str, Any] = {
        "torch": None,
        "cuda": None,
        "python": None,
        "numpy": None,
        "numpy_used": False,
    }
    try:
        saved["torch"] = _torch.get_rng_state()
    except Exception:
        pass
    try:
        if _torch.cuda.is_available():
            saved["cuda"] = _torch.cuda.get_rng_state_all()
    except Exception:
        pass
    try:
        saved["python"] = _random.getstate()
    except Exception:
        pass
    try:
        import numpy as _np

        saved["numpy"] = _np.random.get_state()
        saved["numpy_used"] = True
    except Exception:
        pass
    return saved


def _restore_rng_state(saved: dict[str, Any]) -> None:
    """Restore RNG states (never raises; FP64-reference restore side)."""
    import random as _random

    import torch as _torch

    try:
        if saved.get("torch") is not None:
            _torch.set_rng_state(saved["torch"])
    except Exception:
        pass
    try:
        if saved.get("cuda") is not None and _torch.cuda.is_available():
            _torch.cuda.set_rng_state_all(saved["cuda"])
    except Exception:
        pass
    try:
        if saved.get("python") is not None:
            _random.setstate(saved["python"])
    except Exception:
        pass
    try:
        if saved.get("numpy_used") and saved.get("numpy") is not None:
            import numpy as _np

            _np.random.set_state(saved["numpy"])
    except Exception:
        pass


def _prepare_fp64_clone(eager_bm: Any) -> tuple[Any | None, dict[str, Any]]:
    """Separate genuine FP64 reference owner (reference-only double).

    Deep-copies the whole benchmark (same original module/input/loss/
    optimizer-step semantics -- stock deep-copies for its fp64 golden ref)
    and casts the CLONE's module + floating inputs/aux to float64. The live
    workload benchmark is NEVER touched (the clone is discarded after use);
    only the reference gets double. Callers still save/restore RNG + global
    numerics around the clone's execution. On any failure returns
    (None, fallback-record) following ONLY the established stock cosine
    fallback, explicitly recorded.
    """
    import copy as _copy

    import torch as _torch

    from torch.utils._pytree import tree_map as _tree_map

    record: dict[str, Any] = {}
    try:
        clone = _copy.deepcopy(eager_bm)
    except Exception as error:
        record["fallback"] = "cosine"
        record["fallback_source"] = STOCK_CHECK_SOURCE
        record["fallback_reason"] = (
            f"benchmark deepcopy unsupported ({type(error).__name__}: "
            f"{str(error)[:200]}); FP64 reference cannot be faithfully "
            "obtained -- stock deep-copies for its fp64 golden ref"
        )
        return None, record
    try:
        mod, _ = clone.get_module()
    except Exception as error:
        record["fallback"] = "cosine"
        record["fallback_source"] = STOCK_CHECK_SOURCE
        record["fallback_reason"] = (
            f"clone get_module failed ({type(error).__name__}: "
            f"{str(error)[:200]}); FP64 reference cannot be faithfully obtained"
        )
        return None, record
    try:
        _unwrap_for_state(mod).to(_torch.float64)
    except Exception as error:
        record["fallback"] = "cosine"
        record["fallback_source"] = STOCK_CHECK_SOURCE
        record["fallback_reason"] = (
            f"model cast to float64 unsupported ({type(error).__name__}: "
            f"{str(error)[:200]}); FP64 reference cannot be faithfully obtained"
        )
        return None, record
    # Same original inputs, double (clone only; live inputs never touched).
    for holder in (clone, getattr(clone, "model", None)):
        if holder is None:
            continue
        for attr in ("example_inputs", *_AUX_ATTRS):
            try:
                val = getattr(holder, attr, None)
            except Exception:
                continue
            if val is None:
                continue
            try:
                setattr(
                    holder,
                    attr,
                    _tree_map(
                        lambda x: (
                            x.to(_torch.float64)
                            if isinstance(x, _torch.Tensor) and x.is_floating_point()
                            else x
                        ),
                        val,
                    ),
                )
            except Exception:
                continue
    record["dtype"] = "float64"
    return clone, record


def _fp64_reference_outputs(
    eager_bm: Any, test: str, contexts: Any
) -> tuple[Any | None, dict[str, Any]]:
    """Genuine FP64 diagnostic reference (reference-only double).

    Mirrors env_check.check_accuracy's fp64 golden ref: a SEPARATE deep-copied
    benchmark (same original module/input/driver semantics) cast to float64,
    run through the SAME original driver (eval: benchmark.invoke()), then
    float tensors mapped to float64 exactly like stock. Only the reference
    gets double; the live workload benchmark is NEVER touched (the clone is
    discarded). RNG + global numerics are saved/restored robustly
    (try/finally). If unsupported, follow ONLY the established stock cosine
    fallback (dargs.use_cosine_similarity=True) and record it.
    """
    import torch as _torch

    from torch.utils._pytree import tree_map as _tree_map

    _ = contexts
    record: dict[str, Any] = {"source": STOCK_FP64_SOURCE}
    try:
        incoming = numerical_settings()
    except Exception:
        incoming = {}
    fp32_precision_saved = _save_fp32_precision()
    rng_saved = _snapshot_rng_state()
    try:
        clone, prep = _prepare_fp64_clone(eager_bm)
        record.update(prep)
        if clone is None:
            return None, record
        if test != "eval":
            record["fallback"] = "cosine"
            record["fallback_source"] = STOCK_CHECK_SOURCE
            record["fallback_reason"] = (
                "non-eval reference requested from the eval FP64 helper; "
                "train uses _fp64_train_post_state with the same step callable"
            )
            return None, record
        with _torch.inference_mode():
            fp64_out = clone.invoke()
        fp64_out = _tree_map(
            lambda x: (
                x.to(_torch.float64)
                if isinstance(x, _torch.Tensor) and x.is_floating_point()
                else x
            ),
            fp64_out,
        )
        record["built"] = True
        record["semantics"] = (
            "same-original-driver via separate deep-copied benchmark; "
            "reference-only double"
        )
        return fp64_out, record
    except Exception as error:
        record["fallback"] = "cosine"
        record["fallback_source"] = STOCK_CHECK_SOURCE
        record["fallback_reason"] = (
            f"fp64 execution failed ({type(error).__name__}: {str(error)[:300]}); "
            "stock sets cosine=True"
        )
        return None, record
    finally:
        try:
            _restore_rng_state(rng_saved)
        except Exception:
            pass
        try:
            _restore_global_settings(incoming)
        except Exception:
            pass
        try:
            _restore_fp32_precision(fp32_precision_saved)
        except Exception:
            pass


def _layout_metadata_equal(ref_f: Any, cmp_f: Any) -> tuple[bool, str]:
    """Fail-closed layout/ABI metadata check (no value comparison).

    Verifies dtype/shape/strides/device/sparse/numel for tensor fingerprints,
    dtype/shape for NumPy arrays, key structure for mappings/sequences, and
    exact equality for non-float scalars. Value bytes and float bits are
    SKIPPED here (compared by the stock comparator); anything else failing
    closed. Preserves the existing ABI checks while letting ``same`` judge
    values with the FP64 envelope.
    """
    if isinstance(ref_f, FrozenMapping) or isinstance(cmp_f, FrozenMapping):
        if not isinstance(ref_f, FrozenMapping) or not isinstance(cmp_f, FrozenMapping):
            return False, "mapping-vs-nontensor"
        ref_keys = [k for k, _ in ref_f.pairs]
        cmp_keys = [k for k, _ in cmp_f.pairs]
        if ref_keys != cmp_keys:
            return False, "keys mismatch"
        for (_, rv), (_, cv) in zip(ref_f.pairs, cmp_f.pairs):
            ok, reason = _layout_metadata_equal(rv, cv)
            if not ok:
                return False, reason
        return True, ""
    if isinstance(ref_f, dict) and isinstance(cmp_f, dict):
        ref_is_t = ref_f.get("__tensor__") is True
        cmp_is_t = cmp_f.get("__tensor__") is True
        if ref_is_t or cmp_is_t:
            if not (ref_is_t and cmp_is_t):
                return False, "tensor-vs-nontensor"
            for key in ("dtype", "shape", "strides", "numel", "device", "sparse"):
                if ref_f.get(key) != cmp_f.get(key):
                    return False, f"{key} mismatch (no implicit cast)"
            return True, ""
        if set(ref_f.keys()) == {"__float_bits__"} or set(cmp_f.keys()) == {
            "__float_bits__"
        }:
            if set(ref_f.keys()) != set(cmp_f.keys()):
                return False, "float-vs-nontensor"
            return True, ""
        if "__ndarray__" in ref_f or "__ndarray__" in cmp_f:
            if "__ndarray__" not in ref_f or "__ndarray__" not in cmp_f:
                return False, "ndarray-vs-nontensor"
            rs, cs = ref_f["__ndarray__"], cmp_f["__ndarray__"]
            if rs.get("dtype") != cs.get("dtype") or rs.get("shape") != cs.get("shape"):
                return False, "ndarray dtype/shape mismatch"
            return True, ""
        if "__npscalar__" in ref_f or "__npscalar__" in cmp_f:
            if "__npscalar__" not in ref_f or "__npscalar__" not in cmp_f:
                return False, "npscalar-vs-nontensor"
            if ref_f["__npscalar__"].get("dtype") != cmp_f["__npscalar__"].get("dtype"):
                return False, "npscalar dtype mismatch"
            return True, ""
        if "__seq__" in ref_f or "__seq__" in cmp_f:
            if "__seq__" not in ref_f or "__seq__" not in cmp_f:
                return False, "seq-vs-nontensor"
            if ref_f.get("__tuple__") != cmp_f.get("__tuple__"):
                return False, "tuple-vs-list mismatch"
            rs, cs = ref_f["__seq__"], cmp_f["__seq__"]
            if len(rs) != len(cs):
                return False, "seq length mismatch"
            for rv, cv in zip(rs, cs):
                ok, reason = _layout_metadata_equal(rv, cv)
                if not ok:
                    return False, reason
            return True, ""
        if "__error__" in ref_f or "__error__" in cmp_f:
            return False, "fingerprint_error"
        if set(ref_f.keys()) != set(cmp_f.keys()):
            return False, "keys mismatch"
        for k in sorted(ref_f, key=str):
            ok, reason = _layout_metadata_equal(ref_f[k], cmp_f[k])
            if not ok:
                return False, f"{k}: {reason}"
        return True, ""
    if isinstance(ref_f, (list, tuple)) and isinstance(cmp_f, (list, tuple)):
        if len(ref_f) != len(cmp_f):
            return False, "seq length mismatch"
        for rv, cv in zip(ref_f, cmp_f):
            ok, reason = _layout_metadata_equal(rv, cv)
            if not ok:
                return False, reason
        return True, ""
    if isinstance(ref_f, bytes) or isinstance(cmp_f, bytes):
        # Raw bytes outside tensor payloads: exact (fail closed).
        return (True, "") if ref_f == cmp_f else (False, "bytes differ")
    if isinstance(ref_f, float) or isinstance(cmp_f, float):
        # Live floats reach here only post-thaw edge cases; values are the
        # comparator's job -- metadata passes.
        return True, ""
    # Non-float scalars / None / bools: exact type+value (ABI).
    if type(ref_f) is not type(cmp_f):
        return False, f"type {type(ref_f).__name__} vs {type(cmp_f).__name__}"
    try:
        return (True, "") if bool(ref_f == cmp_f) else (False, "scalar differ")
    except Exception as error:
        return False, type(error).__name__


def _stock_section_pass(
    ref_section: Any,
    cmp_section: Any,
    fp64_section: Any | None,
    *,
    tol: float,
    cos_similarity: bool,
) -> tuple[bool, str]:
    """One section via layout checks + established stock comparator.

    Layout/ABI metadata (dtype/shape/strides/device/keys) fails closed
    without value comparison; values use ``torch._dynamo.utils.same`` with
    the genuine FP64 reference (allclose then FP64 RMSE envelope, cosine
    only per stock policy). Fingerprint-error markers fail closed.
    """
    if _contains_error_marker(ref_section) or _contains_error_marker(cmp_section):
        return False, "fingerprint_error"
    if fp64_section is not None and _contains_error_marker(fp64_section):
        return False, "fp64 fingerprint_error"
    ok_meta, reason_meta = _layout_metadata_equal(ref_section, cmp_section)
    if not ok_meta:
        return False, reason_meta
    try:
        ref_live = _thaw_value(ref_section)
        cmp_live = _thaw_value(cmp_section)
        fp64_live = _thaw_value(fp64_section) if fp64_section is not None else None
    except RestorationError as error:
        return False, f"thaw: {error}"
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"
    ok, detail = _stock_same(
        ref_live, cmp_live, fp64_live, tol=tol, cos_similarity=cos_similarity
    )
    return ok, detail


def _fp64_train_post_state(
    eager_bm: Any, snap_clean: dict[str, Any], step_fn: Any
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Genuine FP64 train post-state (reference-only double, robust restore).

    Same original module/input/loss/optimizer-step semantics via the same
    step callable run on a SEPARATE deep-copied benchmark cast to double;
    the live workload is NEVER touched (no in-place dtype cast, so no
    restore_state dtype-mismatch possible) and stays FP32. RNG + global
    numerics restored in finally. On any failure, follow ONLY the
    established stock cosine fallback and record it.
    """
    record: dict[str, Any] = {"source": STOCK_FP64_SOURCE}
    _ = snap_clean
    try:
        incoming = numerical_settings()
    except Exception:
        incoming = {}
    fp32_precision_saved = _save_fp32_precision()
    rng_saved = _snapshot_rng_state()
    try:
        clone, prep = _prepare_fp64_clone(eager_bm)
        record.update(prep)
        if clone is None:
            return None, record
        _, fp64_fp = step_fn(clone)
        record["built"] = True
        record["semantics"] = (
            "same-original-step via separate deep-copied benchmark; "
            "reference-only double"
        )
        return fp64_fp, record
    except Exception as error:
        record["fallback"] = "cosine"
        record["fallback_source"] = STOCK_CHECK_SOURCE
        record["fallback_reason"] = (
            f"fp64 train step failed ({type(error).__name__}: {str(error)[:300]}); "
            "stock sets cosine=True"
        )
        return None, record
    finally:
        try:
            _restore_rng_state(rng_saved)
        except Exception:
            pass
        try:
            _restore_global_settings(incoming)
        except Exception:
            pass
        try:
            _restore_fp32_precision(fp32_precision_saved)
        except Exception:
            pass


def model_name_of(benchmark: Any) -> str:
    """Best-effort TorchBench model name for stock policy lookup."""
    for attr in ("name", "model_name"):
        try:
            val = getattr(benchmark, attr, None)
        except Exception:
            val = None
        if isinstance(val, str) and val:
            return val
    try:
        inner = getattr(benchmark, "model", None)
        for attr in ("name", "model_name"):
            try:
                val = getattr(inner, attr, None)
            except Exception:
                val = None
            if isinstance(val, str) and val:
                return val
    except Exception:
        pass
    return "unknown"


def eval_accuracy(
    eager_bm: Any, cpp_bm: Any | None, stock_bm: Any | None
) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {}
    with torch.inference_mode():
        ref = eager_bm.invoke()
        ref_clone = _clone_output(ref)
        rep = eager_bm.invoke()
    cmp_self = _compare_outputs(rep, ref, "eager-repeat")
    out["eager_repeat"] = dataclasses.asdict(cmp_self)
    out["eager_repeat_bytewise"] = bool(cmp_self.bytewise_equal)
    # Stock-effective normal acceptance: existing model policy
    # (get_tolerance_and_cosine_flag) + established comparator (same with the
    # genuine FP64 reference). Both CPP and stock compare vs the SAME eager
    # + FP64 reference; stock must be a real valid reference (never masked).
    tb_name = model_name_of(eager_bm)
    try:
        tol, cosine_flag, policy_record = _stock_tolerance_and_flags(tb_name, "eval")
    except Exception as error:
        tol, cosine_flag, policy_record = (
            1e-4,
            False,
            {
                "source": STOCK_TOLICY_SOURCE,
                "error": f"{type(error).__name__}: {error}",
            },
        )
    out["normal_policy"] = policy_record
    out["normal_comparator"] = {"source": STOCK_COMPARATOR_SOURCE}
    fp64_ref, fp64_record = _fp64_reference_outputs(eager_bm, "eval", [])
    out["fp64_reference"] = fp64_record
    effective_cosine = bool(cosine_flag) or fp64_record.get("fallback") == "cosine"
    if effective_cosine and fp64_record.get("fallback") == "cosine":
        out["normal_policy"]["cosine_fallback"] = fp64_record.get("fallback_reason")
        out["normal_policy"]["cosine_fallback_source"] = fp64_record.get(
            "fallback_source"
        )
    out["normal_tolerance"] = {"tol": tol, "cosine": effective_cosine}
    if cpp_bm is None:
        out["cpp_vs_eager"] = {
            "error": "cpp variant unavailable (compile failed); never compared against eager itself"
        }
        out["cpp_vs_eager_normal_tol_pass"] = False
        out["cpp_vs_eager_stock_pass"] = False
        out["cpp_vs_stock"] = {"error": "cpp variant unavailable"}
    else:
        with torch.inference_mode():
            cpp_out = cpp_bm.invoke()
        cmp_cpp = _compare_outputs(cpp_out, ref_clone, "cpp-vs-eager")
        out["cpp_vs_eager"] = dataclasses.asdict(cmp_cpp)
    # Legacy assert_close diagnostic kept for audit (NOT acceptance).
    if cpp_bm is None:
        pass
    else:
        try:
            torch.testing.assert_close(
                cpp_out,
                ref_clone,
                rtol=NORMAL_RTOL,
                atol=NORMAL_ATOL,
                check_stride=True,
            )
            out["cpp_vs_eager_normal_tol_pass"] = True
            out["cpp_vs_eager_normal_tol"] = {"rtol": NORMAL_RTOL, "atol": NORMAL_ATOL}
        except Exception as error:
            out["cpp_vs_eager_normal_tol_pass"] = False
            out["cpp_vs_eager_normal_tol_error"] = (
                f"{type(error).__name__}: {str(error)[:1000]}"
            )
        ok, detail = _stock_same(
            ref_clone, cpp_out, fp64_ref, tol=tol, cos_similarity=effective_cosine
        )
        out["cpp_vs_eager_stock_pass"] = bool(ok)
        out["cpp_vs_eager_stock_detail"] = detail
    stock_out: Any = None
    stock_ok: bool | None = None
    if stock_bm is not None:
        try:
            with torch.inference_mode():
                stock_out = stock_bm.invoke()
            cmp_stock = _compare_outputs(stock_out, ref_clone, "stock-vs-eager")
            out["stock_vs_eager"] = dataclasses.asdict(cmp_stock)
            try:
                torch.testing.assert_close(
                    stock_out,
                    ref_clone,
                    rtol=NORMAL_RTOL,
                    atol=NORMAL_ATOL,
                    check_stride=True,
                )
                out["stock_vs_eager_normal_tol_pass"] = True
            except Exception as error:
                out["stock_vs_eager_normal_tol_pass"] = False
                out["stock_vs_eager_normal_tol_error"] = (
                    f"{type(error).__name__}: {str(error)[:1000]}"
                )
            ok_s, detail_s = _stock_same(
                ref_clone,
                stock_out,
                fp64_ref,
                tol=tol,
                cos_similarity=effective_cosine,
            )
            stock_ok = bool(ok_s)
            out["stock_vs_eager_stock_pass"] = stock_ok
            out["stock_vs_eager_stock_detail"] = detail_s
        except Exception as error:
            out["stock_vs_eager"] = {"error": f"{type(error).__name__}: {error}"}
            out["stock_vs_eager_stock_pass"] = False
    else:
        out["stock_vs_eager_stock_pass"] = None
    # CPP-vs-stock diagnostic (never acceptance): same stock comparator
    # between the two compiled variants with the identical FP64 reference.
    if cpp_bm is not None and stock_out is not None:
        try:
            ok_cs, detail_cs = _stock_same(
                stock_out, cpp_out, fp64_ref, tol=tol, cos_similarity=effective_cosine
            )
            out["cpp_vs_stock"] = {"stock_pass": bool(ok_cs), "detail": detail_cs}
        except Exception as error:
            out["cpp_vs_stock"] = {"error": f"{type(error).__name__}: {error}"}
    # Require a real valid stock reference: never mask a stock failure and
    # never claim performance on an unproven reference.
    out["stock_reference_valid"] = stock_ok if stock_ok is not None else None
    return out


def _freeze_stateless_scalar(value: Any) -> Any:
    """Freeze a non-tensor loss leaf (static scalar) without fabrication."""
    if value is None:
        return None
    try:
        return _freeze_value(value)
    except Exception:
        return {"__error__": f"non-tensor loss of type {type(value).__name__}"}


def _post_state_fingerprint(benchmark: Any, loss: Any) -> dict[str, Any]:
    """Frozen post-step fingerprint (memory only; never into the scorecard).

    Covers params, buffers, grads, loss, optimizer state, schedulers, and
    torch/CUDA/Python/NumPy RNG. Only diff *metadata* is reported, so no
    DLRM-sized payload can land in the JSON report.
    """
    import torch

    fp: dict[str, Any] = {}
    try:
        mod, _ = benchmark.get_module()
    except Exception as error:
        return {"__error__": f"get_module: {type(error).__name__}: {error}"}
    target = _unwrap_for_state(mod)
    try:
        params: dict[str, Any] = {}
        for k, v in target.state_dict().items():
            if isinstance(v, torch.Tensor):
                params[k] = _tensor_fingerprint(v)
            else:
                try:
                    params[k] = _freeze_value(v)
                except Exception as error:
                    params[k] = {"__error__": f"{type(error).__name__}: {error}"}
        fp["params"] = params
    except Exception as error:
        fp["params_error"] = f"{type(error).__name__}: {error}"
    try:
        buffers: dict[str, Any] = {}
        for name, buf in target.named_buffers():
            if isinstance(buf, torch.Tensor):
                buffers[name] = _tensor_fingerprint(buf)
            else:
                try:
                    buffers[name] = _freeze_value(buf)
                except Exception as error:
                    buffers[name] = {"__error__": f"{type(error).__name__}: {error}"}
        fp["buffers"] = buffers
    except Exception as error:
        fp["buffers_error"] = f"{type(error).__name__}: {error}"
    try:
        grads: dict[str, Any] = {}
        for name, p in target.named_parameters():
            g = p.grad
            if g is None:
                grads[name] = None
            elif isinstance(g, torch.Tensor):
                grads[name] = _tensor_fingerprint(g)
            else:
                grads[name] = {"__error__": f"non-tensor grad {type(g).__name__}"}
        fp["grads"] = grads
    except Exception as error:
        fp["grads_error"] = f"{type(error).__name__}: {error}"
    try:
        # Exactly one typed loss fingerprint (fix 1): _one_step already
        # produces a typed fingerprint (tensor) or frozen scalar (stateless).
        # An already-fingerprinted loss passes through untouched -- never
        # re-frozen (re-freezing would wrap bytes into a FrozenMapping and
        # force exact comparison, defeating normal tolerance).
        if isinstance(loss, torch.Tensor):
            fp["loss"] = _tensor_fingerprint(loss)
        elif loss is None:
            fp["loss"] = None
        elif isinstance(loss, (FrozenMapping, FrozenKey)):
            fp["loss"] = loss
        elif isinstance(loss, dict) and (
            loss.get("__tensor__") is True
            or "__float_bits__" in loss
            or "__error__" in loss
            or "__ndarray__" in loss
            or "__npscalar__" in loss
        ):
            fp["loss"] = loss
        else:
            try:
                fp["loss"] = _freeze_value(loss)
            except Exception as error:
                fp["loss"] = {"__error__": f"{type(error).__name__}: {error}"}
    except Exception as error:
        fp["loss"] = {"__error__": f"{type(error).__name__}: {error}"}
    try:
        opt = benchmark.get_optimizer()
        fp["has_optimizer"] = opt is not None
        if opt is not None:
            try:
                fp["optimizer_state"] = _freeze_value(opt.state_dict())
            except Exception as error:
                fp["optimizer_state"] = {
                    "__error__": f"{type(error).__name__}: {error}"
                }
    except Exception as error:
        fp["optimizer_error"] = f"{type(error).__name__}: {error}"
    try:
        fp["schedulers"] = {
            name: _freeze_scheduler(sched)
            for name, sched in _find_schedulers(benchmark).items()
        }
    except Exception as error:
        fp["schedulers"] = {"__error__": f"{type(error).__name__}: {error}"}
    try:
        fp["torch_rng"] = _tensor_fingerprint(torch.get_rng_state())
    except Exception as error:
        fp["torch_rng"] = {"__error__": f"{type(error).__name__}: {error}"}
    try:
        if torch.cuda.is_available():
            fp["cuda_rng"] = [
                _tensor_fingerprint(s) for s in torch.cuda.get_rng_state_all()
            ]
        else:
            fp["cuda_rng"] = []
    except Exception as error:
        fp["cuda_rng"] = {"__error__": f"{type(error).__name__}: {error}"}
    try:
        fp["python_rng"] = {
            "__python_rng__": {"repr": repr(random.getstate())},
            "state": _freeze_value(random.getstate()),
        }
    except Exception as error:
        fp["python_rng"] = {"__error__": f"{type(error).__name__}: {error}"}
    try:
        import numpy as _np

        state = _np.random.get_state()
        fp["numpy_rng"] = {
            "__numpy_rng__": {"algorithm": str(state[0])},
            "state": _freeze_value(state),
        }
    except Exception:
        fp["numpy_rng"] = {"__error__": "numpy unavailable"}
    return fp


def _tensor_fingerprint(tensor: Any) -> dict[str, Any]:
    """Typed tensor fingerprint: dtype+shape+strides metadata with raw bytes.

    Bytes are the contiguous logical values; strides are recorded separately
    so layout is never erased. Fingerprints live in memory only and are never
    serialized into the JSON scorecard (only diff metadata is reported, so no
    DLRM-sized payload can land in the report).
    """
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            "loops_ir suite tensor fingerprint requires a torch.Tensor, got "
            f"{type(tensor).__name__}"
        )
    try:
        shape = [int(s) for s in tensor.shape]
        strides = [int(s) for s in tensor.stride()]
        blob = _raw_bytes(tensor)
    except Exception as error:
        # Sparse grads (e.g. DLRM embedding_bag) have no dense layout:
        # fingerprint indices/values explicitly instead of failing.
        if tensor.is_sparse or getattr(tensor, "is_sparse_csr", False):
            try:
                dense = tensor.to_dense()
                return {
                    "__tensor__": True,
                    "dtype": str(tensor.dtype),
                    "shape": [int(s) for s in tensor.shape],
                    "strides": None,
                    "numel": int(tensor.numel()),
                    "device": str(tensor.device),
                    "sparse": True,
                    "bytes": _raw_bytes(dense),
                }
            except Exception as inner:
                raise TypeError(
                    "loops_ir suite cannot fingerprint sparse tensor "
                    f"({type(inner).__name__}: {inner})"
                ) from inner
        raise TypeError(
            "loops_ir suite cannot fingerprint tensor "
            f"({type(error).__name__}: {error})"
        ) from error
    return {
        "__tensor__": True,
        "dtype": str(tensor.dtype),
        "shape": shape,
        "strides": strides,
        "numel": int(tensor.numel()),
        "device": str(tensor.device),
        "bytes": blob,
    }


def _freeze_value(value: Any) -> Any:
    """Freeze an arbitrary state leaf into comparable typed data.

    Preserves every typed scalar: floats as raw bits (keeps -0.0 and NaN
    payloads), ints/strings/None natively (bools stay bools and never equal
    ints), containers structurally, tensors via :func:`_tensor_fingerprint`.
    Raises TypeError for unsupported types -- fail closed, never an invented
    None placeholder that would hide a dropped LR/momentum/counter.
    """
    import struct

    import torch

    if value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, bytes):
        # Raw fingerprint bytes (tensor blobs, NumPy uint32 state): opaque
        # but exactly comparable; pass through, never decode/reinterpret.
        return value
    if isinstance(value, float):
        return {"__float_bits__": struct.pack(">d", value).hex()}
    if isinstance(value, torch.Tensor):
        return _tensor_fingerprint(value)
    try:
        import numpy as _np

        if isinstance(value, _np.ndarray):
            return {
                "__ndarray__": {
                    "dtype": str(value.dtype),
                    "shape": [int(s) for s in value.shape],
                    "bytes": bytes(value.tobytes()),
                }
            }
        if isinstance(value, _np.generic):
            return {
                "__npscalar__": {
                    "dtype": str(value.dtype),
                    "bytes": bytes(_np.ascontiguousarray(value).tobytes()),
                }
            }
    except ImportError:
        pass
    if isinstance(value, dict):
        # Exact key types via FrozenMapping (finding 2): int Adam-state
        # keys (0, 1, ...) must round-trip as ints, never str('0').
        try:
            return _freeze_mapping(value)
        except TypeError:
            raise
        except Exception as error:
            raise TypeError(
                "loops_ir suite cannot freeze mapping "
                f"({type(error).__name__}: {error})"
            ) from error
    if isinstance(value, (list, tuple)):
        return {
            "__seq__": [_freeze_value(v) for v in value],
            "__tuple__": isinstance(value, tuple),
        }
    raise TypeError(
        "loops_ir suite cannot fingerprint state of type "
        f"{type(value).__name__}; failing closed (no invented placeholder)"
    )


class FrozenKey(NamedTuple):
    """Exact dict-key preservation (finding 2).

    ``kind`` is one of "none"/"bool"/"int"/"floatbits"/"str"/"tuple";
    ``payload`` carries the value (floats as raw-bits hex, tuples as a
    tuple of FrozenKey). A NamedTuple (not a dataclass) so the class works
    under any import mechanism. Python dicts already collapse 1/True/1.0
    at runtime, so no magic is attempted there -- but a frozen round-trip
    never invents a str('0') for int 0 or vice versa.
    """

    kind: str
    payload: Any = None


class FrozenMapping(NamedTuple):
    """A mapping with exact key types (finding 2).

    A NamedTuple -- never a plain dict -- so the marker fields can never
    collide with genuine user keys like ``__float_bits__``. Plain
    ``__dict__``/``__keys__`` markers are retired.
    """

    pairs: tuple = ()


def _freeze_key(key: Any) -> FrozenKey:
    import struct

    if key is None:
        return FrozenKey("none")
    if isinstance(key, bool):
        return FrozenKey("bool", bool(key))
    if isinstance(key, int):
        return FrozenKey("int", int(key))
    if isinstance(key, float):
        return FrozenKey("floatbits", struct.pack(">d", key).hex())
    if isinstance(key, str):
        return FrozenKey("str", key)
    if isinstance(key, tuple):
        return FrozenKey("tuple", tuple(_freeze_key(k) for k in key))
    raise TypeError(
        "loops_ir suite cannot freeze dict key of type "
        f"{type(key).__name__}; failing closed"
    )


def _thaw_key(frozen: Any) -> Any:
    import struct

    if not isinstance(frozen, FrozenKey):
        raise RestorationError(
            f"mapping key is not a FrozenKey ({type(frozen).__name__}); "
            "no hidden defaults"
        )
    if frozen.kind == "none":
        return None
    if frozen.kind == "bool":
        return bool(frozen.payload)
    if frozen.kind == "int":
        return int(frozen.payload)
    if frozen.kind == "floatbits":
        try:
            return struct.unpack(">d", bytes.fromhex(frozen.payload))[0]
        except Exception as error:
            raise RestorationError(
                f"thaw float key ({type(error).__name__}: {error})"
            ) from error
    if frozen.kind == "str":
        return str(frozen.payload)
    if frozen.kind == "tuple":
        try:
            return tuple(_thaw_key(k) for k in frozen.payload)
        except RestorationError:
            raise
        except Exception as error:
            raise RestorationError(
                f"thaw tuple key ({type(error).__name__}: {error})"
            ) from error
    raise RestorationError(f"unknown FrozenKey kind {frozen.kind!r}")


def _freeze_mapping(mapping: Any) -> FrozenMapping:
    try:
        items = list(mapping.items())
    except Exception as error:
        raise TypeError(
            "loops_ir suite cannot freeze non-mapping "
            f"({type(error).__name__}: {error})"
        ) from error
    pairs = []
    for k, v in items:
        pairs.append((_freeze_key(k), _freeze_value(v)))
    return FrozenMapping(pairs=tuple(pairs))


def _thaw_mapping(frozen: Any) -> Any:
    if not isinstance(frozen, FrozenMapping):
        raise RestorationError(
            f"mapping is not a FrozenMapping ({type(frozen).__name__})"
        )
    out: dict[Any, Any] = {}
    for k, v in frozen.pairs:
        out[_thaw_key(k)] = _thaw_value(v)
    return out


def _frozen_mapping_equal(a: Any, b: Any) -> bool:
    if not isinstance(a, FrozenMapping) or not isinstance(b, FrozenMapping):
        return False
    if len(a.pairs) != len(b.pairs):
        return False
    for (ka, va), (kb, vb) in zip(a.pairs, b.pairs):
        if ka != kb:
            return False
        if not _frozen_equal(va, vb):
            return False
    return True


def _frozen_equal(a: Any, b: Any) -> bool:
    """Exact structural equality over frozen values (strict lane)."""
    if isinstance(a, FrozenMapping) or isinstance(b, FrozenMapping):
        return _frozen_mapping_equal(a, b)
    if isinstance(a, FrozenKey) or isinstance(b, FrozenKey):
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_frozen_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_frozen_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, bytes) and isinstance(b, bytes):
        return a == b
    if type(a) is not type(b):
        return False
    try:
        return bool(a == b)
    except Exception:
        return False


def _contains_error_marker(value: Any) -> bool:
    if isinstance(value, FrozenMapping):
        return any(_contains_error_marker(v) for _, v in value.pairs)
    if isinstance(value, dict):
        if "__error__" in value:
            return True
        return any(_contains_error_marker(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_error_marker(v) for v in value)
    return False


def _diff_paths(section: str, a: Any, b: Any, cap: int = 100) -> list[str]:
    """Summarize differing leaf paths as metadata (never raw values)."""
    out: list[str] = []

    def _walk(path: str, x: Any, y: Any) -> None:
        if len(out) >= cap:
            return
        if isinstance(x, FrozenMapping) or isinstance(y, FrozenMapping):
            if not isinstance(x, FrozenMapping) or not isinstance(y, FrozenMapping):
                out.append(f"{path}:mapping-vs-nontensor")
                return
            if [k for k, _ in x.pairs] != [k for k, _ in y.pairs]:
                out.append(f"{path}:keys")
                return
            for (k, vx), (_, vy) in zip(x.pairs, y.pairs):
                _walk(f"{path}.{k.payload}", vx, vy)
            return
        if isinstance(x, dict) and isinstance(y, dict):
            if set(x.keys()) != set(y.keys()):
                out.append(f"{path}:keys")
                return
            for k in sorted(x, key=str):
                _walk(f"{path}.{k}", x[k], y[k])
            return
        if _frozen_equal(x, y):
            return
        out.append(path)

    _walk(section, a, b)
    return out


_FINGERPRINT_SECTIONS = (
    "params",
    "buffers",
    "grads",
    "loss",
    "optimizer_state",
    "schedulers",
    "torch_rng",
    "cuda_rng",
    "python_rng",
    "numpy_rng",
)


def _fingerprints_equal(a: dict[str, Any], b: dict[str, Any]) -> tuple[bool, list[str]]:
    """Strict post-state equality: every section bytewise/layout-exact.

    Any fingerprint-error marker on either side is a diff (a section that
    could not be captured proves nothing). Returns (equal, diff-metadata).
    """
    diffs: list[str] = []
    for section in _FINGERPRINT_SECTIONS:
        va = a.get(section, "__missing__")
        vb = b.get(section, "__missing__")
        if va == "__missing__" or vb == "__missing__":
            if va != vb:
                diffs.append(f"{section}.missing")
            continue
        if _contains_error_marker(va) or _contains_error_marker(vb):
            diffs.append(f"{section}.fingerprint_error")
            continue
        if not _frozen_equal(va, vb):
            diffs.extend(_diff_paths(section, va, vb))
    return (len(diffs) == 0, diffs)


def _bits_to_float(hexstr: str) -> float:
    import struct

    return struct.unpack(">d", bytes.fromhex(hexstr))[0]


def _reconstruct_tensor(frozen: Any) -> Any:
    """Rebuild a tensor from a typed fingerprint for normal-lane closeness.

    Dtype/shape/strides are verified equal by the caller first (never an
    implicit cast). The payload is the COMPACT CONTIGUOUS logical values, so
    decoding is ``view(dtype).reshape(shape)`` -- never ``as_strided`` over
    the compact blob (that reads out of bounds for strided originals, e.g.
    shape 8x4 with strides 8x2 over 32 stored values). The recorded strides
    stay metadata for equality; reconstructed tensors compare by logical
    value. Empty and 0-dim tensors return typed empties/scalars.
    """
    import torch

    if not isinstance(frozen, dict) or frozen.get("__tensor__") is not True:
        raise TypeError("normal-lane reconstruct requires a tensor fingerprint")
    dtype = getattr(torch, str(frozen["dtype"]).split(".")[-1], None)
    if dtype is None:
        raise TypeError(f"unknown dtype {frozen.get('dtype')}")
    blob = frozen.get("bytes")
    if not isinstance(blob, bytes):
        raise TypeError("tensor fingerprint bytes missing")
    shape = tuple(int(s) for s in frozen.get("shape", ()))
    numel = int(frozen.get("numel", -1))
    device = frozen.get("device", None)
    if numel == 0:
        try:
            return torch.empty(shape, dtype=dtype, device=device or "cpu")
        except Exception:
            return torch.empty(shape, dtype=dtype)
    try:
        elems = torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
    except Exception as error:
        raise TypeError(f"byte payload unreadable: {error}") from error
    try:
        flat = elems.view(dtype)
    except Exception as error:
        raise TypeError(f"byte blob does not match dtype {dtype}: {error}") from error
    if flat.numel() != numel:
        raise TypeError(
            f"byte blob holds {flat.numel()} elements, fingerprint says {numel}"
        )
    try:
        out = flat.reshape(shape)
    except Exception as error:
        raise TypeError(
            f"cannot reshape {flat.numel()} values to {shape}: {error}"
        ) from error
    # Placement restores the recorded device (aux inputs must return to
    # CUDA, not silently CPU). Missing key -> CPU for backward compat;
    # unparseable device -> explicit error, never silent CPU.
    if device is None:
        return out
    try:
        return out.to(device)
    except Exception as error:
        raise TypeError(f"cannot place tensor on {device}: {error}") from error


def _normal_tensor_close(
    ref_f: Any, cmp_f: Any, *, rtol: float, atol: float
) -> tuple[bool, str]:
    """Quantitative tensor closeness with an explicit contract (normal lane).

    Fails closed on dtype/shape/stride mismatch (never an implicit cast);
    otherwise ``torch.testing.assert_close`` with the public recorded
    tolerances. Returns (ok, reason).
    """
    import torch

    if not isinstance(ref_f, dict) or not isinstance(cmp_f, dict):
        return False, "non-fingerprint leaf"
    for key in ("dtype", "shape"):
        if ref_f.get(key) != cmp_f.get(key):
            return False, f"{key} mismatch (no implicit cast)"
    if (ref_f.get("sparse") is True) != (cmp_f.get("sparse") is True):
        return False, "sparse-vs-dense mismatch"
    if ref_f.get("strides") != cmp_f.get("strides"):
        return False, "strides mismatch (no implicit layout change)"
    if ref_f.get("device", None) != cmp_f.get("device", None):
        # Placement mismatch fails closed (a CPU tensor must never pass
        # for a CUDA reference, even with equal values).
        return (
            False,
            f"device mismatch ({ref_f.get('device')} vs {cmp_f.get('device')})",
        )
    try:
        ref_t = _reconstruct_tensor(ref_f)
        cmp_t = _reconstruct_tensor(cmp_f)
    except Exception as error:
        return False, f"reconstruct: {type(error).__name__}: {error}"
    try:
        torch.testing.assert_close(cmp_t, ref_t, rtol=rtol, atol=atol)
        return True, ""
    except AssertionError as error:
        return False, str(error)[:300]
    except Exception as error:
        return False, f"{type(error).__name__}: {str(error)[:300]}"


def _normal_frozen_close(
    ref_f: Any, cmp_f: Any, *, rtol: float, atol: float
) -> tuple[bool, str]:
    """Recursive normal-lane closeness over frozen state.

    Tensors compare by tolerance (via :func:`_normal_tensor_close`), frozen
    float bits compare quantitatively, structure/scalars compare exactly.
    Returns (ok, reason).
    """
    if isinstance(ref_f, FrozenMapping) or isinstance(cmp_f, FrozenMapping):
        if not isinstance(ref_f, FrozenMapping) or not isinstance(cmp_f, FrozenMapping):
            return False, "mapping-vs-nontensor"
        if [k for k, _ in ref_f.pairs] != [k for k, _ in cmp_f.pairs]:
            return False, "keys mismatch"
        for (k, rv), (_, cv) in zip(ref_f.pairs, cmp_f.pairs):
            ok, reason = _normal_frozen_close(rv, cv, rtol=rtol, atol=atol)
            if not ok:
                return False, f"{k.payload}: {reason}"
        return True, ""
    if isinstance(ref_f, dict) and isinstance(cmp_f, dict):
        if ref_f.get("__tensor__") is True or cmp_f.get("__tensor__") is True:
            if (
                ref_f.get("__tensor__") is not True
                or cmp_f.get("__tensor__") is not True
            ):
                return False, "tensor-vs-nontensor"
            return _normal_tensor_close(ref_f, cmp_f, rtol=rtol, atol=atol)
        if set(ref_f.keys()) != set(cmp_f.keys()):
            return False, "keys mismatch"
        if set(ref_f.keys()) == {"__float_bits__"}:
            try:
                ra = _bits_to_float(ref_f["__float_bits__"])
                rb = _bits_to_float(cmp_f["__float_bits__"])
            except Exception as error:
                return False, f"float bits: {error}"
            if ra == rb:
                return True, ""
            tol = atol + rtol * max(abs(ra), abs(rb))
            if abs(ra - rb) <= tol:
                return True, ""
            return False, f"float {ra} vs {rb} outside tol"
        for k in sorted(ref_f, key=str):
            ok, reason = _normal_frozen_close(ref_f[k], cmp_f[k], rtol=rtol, atol=atol)
            if not ok:
                return False, f"{k}: {reason}"
        return True, ""
    if type(ref_f) is not type(cmp_f):
        return False, f"type {type(ref_f).__name__} vs {type(cmp_f).__name__}"
    if isinstance(ref_f, bytes):
        return (True, "") if ref_f == cmp_f else (False, "bytes differ")
    try:
        return (True, "") if bool(ref_f == cmp_f) else (False, "scalar differ")
    except Exception as error:
        return False, f"{type(error).__name__}"


def _normal_section_close(
    ref_section: Any, cmp_section: Any, *, rtol: float, atol: float
) -> tuple[bool, list[str]]:
    """Normal-lane closeness for one named state section.

    Fingerprint-error markers fail closed (an uncaptured section proves
    nothing). A section that IS itself a tensor fingerprint (e.g. loss)
    delegates whole to the tensor comparison -- never iterated as
    metadata/bytes leaves. Returns (ok, diff-metadata).
    """
    if _contains_error_marker(ref_section) or _contains_error_marker(cmp_section):
        return False, ["fingerprint_error"]
    if isinstance(ref_section, dict) and ref_section.get("__tensor__") is True:
        if (
            not isinstance(cmp_section, dict)
            or cmp_section.get("__tensor__") is not True
        ):
            return False, ["tensor-vs-nontensor"]
        ok, reason = _normal_tensor_close(
            ref_section, cmp_section, rtol=rtol, atol=atol
        )
        return (ok, [] if ok else [reason])
    if not isinstance(ref_section, dict) or not isinstance(cmp_section, dict):
        ok, reason = _normal_frozen_close(
            ref_section, cmp_section, rtol=rtol, atol=atol
        )
        return (ok, [] if ok else [reason])
    diffs: list[str] = []
    for key in sorted(set(ref_section) | set(cmp_section), key=str):
        if key not in ref_section or key not in cmp_section:
            diffs.append(f"{key}:missing")
            continue
        ok, reason = _normal_frozen_close(
            ref_section[key], cmp_section[key], rtol=rtol, atol=atol
        )
        if not ok:
            diffs.append(f"{key}: {reason}")
    return (len(diffs) == 0, diffs)


def _freeze_scheduler(sched: Any) -> Any:
    """Explicit scheduler post-state fingerprint.

    Covers state_dict schedulers (e.g. LRPolicyScheduler) and plain-attribute
    schedulers (e.g. BERT's ScheduledOptim: every numeric attr plus the inner
    Adam state). Returns an error marker on failure, never silence.
    """
    try:
        if hasattr(sched, "state_dict"):
            try:
                return {
                    "kind": "state_dict",
                    "state": _freeze_value(sched.state_dict()),
                }
            except Exception as error:
                return {
                    "__error__": f"scheduler state_dict: {type(error).__name__}: {error}"
                }
        try:
            names = sorted(str(k) for k in vars(sched))
        except Exception as error:
            return {"__error__": f"scheduler vars: {type(error).__name__}: {error}"}
        frozen_attrs: dict[str, Any] = {}
        for name in names:
            if name == "_optimizer":
                continue
            try:
                frozen_attrs[name] = _freeze_value(getattr(sched, name))
            except Exception as error:
                frozen_attrs[name] = {"__error__": f"{type(error).__name__}: {error}"}
        inner = getattr(sched, "_optimizer", None)
        inner_state: Any = None
        if inner is not None and hasattr(inner, "state_dict"):
            try:
                inner_state = _freeze_value(inner.state_dict())
            except Exception as error:
                inner_state = {
                    "__error__": f"inner optimizer: {type(error).__name__}: {error}"
                }
        return {
            "kind": "plain_attrs",
            "attrs": frozen_attrs,
            "inner_optimizer": inner_state,
        }
    except Exception as error:
        return {"__error__": f"scheduler: {type(error).__name__}: {error}"}


def _unwrap_for_state(module: Any) -> Any:
    """Unwrap torch.compile's module wrapper for state_dict key stability.

    A compiled module may expose ``_orig_mod``-prefixed keys; restoring into
    the unwrapped original keeps canonical eager key names while preserving
    Parameter object identity (values copy in place, never rebind).
    """
    try:
        import torch

        inner = getattr(module, "_orig_mod", None)
        if isinstance(inner, torch.nn.Module):
            return inner
    except Exception:
        pass
    return module


def train_accuracy(
    eager_bm: Any, cpp_bm: Any | None, stock_bm: Any | None
) -> dict[str, Any]:
    """Strict bytewise post-state comparison after state restoration.

    Uses genuine staged ``forward``/``backward``/``optimizer_step`` where the
    model is staged, else the genuine ``train()`` with a temporary native
    loss observer (removed before timing). Never fabricates a loss.
    ``cpp_bm=None`` (failed compile) records an explicit error, never a
    self-comparison.
    """
    out: dict[str, Any] = {"staged": _is_staged(eager_bm)}

    def _one_step(bm: Any) -> tuple[Any, dict[str, Any]]:
        from torchbenchmark.util.model import nested as _tb_nested

        if _is_staged(bm):
            opt = bm.get_optimizer()
            with _tb_nested(*bm.run_contexts):
                if opt is not None:
                    opt.zero_grad()
                with _tb_nested(*getattr(bm, "forward_contexts", [])):
                    loss = bm.forward()
                loss_clone = (
                    _tensor_fingerprint(loss)
                    if hasattr(loss, "detach")
                    else _freeze_stateless_scalar(loss)
                )
                with _tb_nested(*getattr(bm, "backward_contexts", [])):
                    bm.backward(loss)
                if opt is not None:
                    with _tb_nested(*getattr(bm, "optimizer_contexts", [])):
                        bm.optimizer_step()
            return loss_clone, _post_state_fingerprint(bm, loss_clone)
        # Non-staged (custom BERT / DLRM / Timm): drive the genuine full
        # step through bm.invoke() so run_contexts (pick_grad, profiling
        # executor) apply exactly as in production. The loss observer uses
        # the original native loss (ScheduledOptim/NLL, LRPolicy/MSE/SGD,
        # timm SGD/CrossEntropy with its random target) and is removed
        # before timing. Never a substitute NN loss.
        with _LossObserver(bm) as obs:
            bm.invoke()
            raw = obs.last_loss
            if hasattr(raw, "detach"):
                loss_clone = _tensor_fingerprint(raw)
            else:
                loss_clone = _freeze_stateless_scalar(raw)
            via = obs.via
        return loss_clone, {**_post_state_fingerprint(bm, loss_clone), "loss_via": via}

    # Eager repeat stability from the same restored state.
    snap_clean = snapshot_state(eager_bm)
    try:
        restore_state(eager_bm, snap_clean)
        _, fp1 = _one_step(eager_bm)
        draws1 = _continuation_draws()
        restore_state(eager_bm, snap_clean)
        _, fp2 = _one_step(eager_bm)
        draws2 = _continuation_draws()
    except RestorationError as error:
        return {
            "staged": _is_staged(eager_bm),
            "error_type": "RestorationError",
            "error": f"eager repeat restore: {error}",
        }
    except Exception as error:
        return {
            "staged": _is_staged(eager_bm),
            "error_type": type(error).__name__,
            "error": f"eager repeat step: {error}",
        }
    stable, stable_diffs = _fingerprints_equal(fp1, fp2)
    draws_stable, draws_diffs = _fingerprints_equal(
        {"draws": draws1}, {"draws": draws2}
    )
    if not draws_stable:
        stable = False
        stable_diffs = stable_diffs + [f"rng_draws.{d}" for d in draws_diffs[:20]]
    out["eager_repeat_stable"] = bool(stable)
    out["eager_repeat_diffs"] = stable_diffs[:50]
    if "loss_via" in fp1:
        out["loss_via"] = fp1["loss_via"]
        out["loss_available"] = fp1.get("loss") is not None
    else:
        out["loss_via"] = "staged-forward"
        out["loss_available"] = fp1.get("loss") is not None
    if not stable:
        out["note"] = (
            "eager is not repeat-stable under the original profile; strict "
            "rejects the original profile (no hidden deterministic flags). A "
            "separately declared matched deterministic validation profile may "
            "be recorded explicitly."
        )
    # Reference post-state from clean.
    try:
        restore_state(eager_bm, snap_clean)
        ref_loss, ref_fp = _one_step(eager_bm)
        ref_draws = _continuation_draws()
    except RestorationError as error:
        return {
            "staged": out.get("staged"),
            "eager_repeat_stable": out.get("eager_repeat_stable"),
            "eager_repeat_diffs": out.get("eager_repeat_diffs", []),
            "error_type": "RestorationError",
            "error": f"eager reference restore: {error}",
        }
    except Exception as error:
        return {
            "staged": out.get("staged"),
            "eager_repeat_stable": out.get("eager_repeat_stable"),
            "eager_repeat_diffs": out.get("eager_repeat_diffs", []),
            "error_type": type(error).__name__,
            "error": f"eager reference step: {error}",
        }
    out["reference_loss_available"] = ref_loss is not None
    if cpp_bm is None:
        out["cpp_vs_eager"] = {
            "error": "cpp variant unavailable (compile failed); never compared against eager itself"
        }
        out["cpp_vs_eager_strict_equal"] = False
        out["cpp_vs_eager_diffs"] = ["cpp-unavailable"]
        out["cpp_normal_full_state_pass"] = False
        out["cpp_normal_full_state_diffs"] = ["cpp-unavailable"]
        out["cpp_loss_normal_tol_pass"] = False
        out["fp64_reference"] = {
            "source": STOCK_FP64_SOURCE,
            "built": False,
            "note": "not attempted: cpp unavailable",
        }
        out["stock_reference_valid"] = None
        return out
    # CPP post-state from the same clean state.
    try:
        snap_cpp = snapshot_state(cpp_bm)
        # Align CPP start state to the eager clean state (same values).
        restore_state(cpp_bm, snap_clean)
    except RestorationError as error:
        out["cpp_vs_eager"] = {
            "error_type": "RestorationError",
            "error": f"cpp align restore: {error}",
        }
        out["cpp_vs_eager_strict_equal"] = False
        out["cpp_normal_full_state_pass"] = False
        out["cpp_normal_full_state_diffs"] = ["cpp-align-restore-veto"]
        out["cpp_loss_normal_tol_pass"] = False
        out["fp64_reference"] = {
            "source": STOCK_FP64_SOURCE,
            "built": False,
            "note": "not attempted: cpp align restore veto",
        }
        out["stock_reference_valid"] = None
        return out
    except Exception as error:
        out["cpp_vs_eager"] = {"error": f"{type(error).__name__}: {error}"}
        out["cpp_vs_eager_strict_equal"] = False
        out["cpp_normal_full_state_pass"] = False
        out["cpp_normal_full_state_diffs"] = ["cpp-align-restore-veto"]
        out["cpp_loss_normal_tol_pass"] = False
        out["fp64_reference"] = {
            "source": STOCK_FP64_SOURCE,
            "built": False,
            "note": "not attempted: cpp align restore veto",
        }
        out["stock_reference_valid"] = None
        return out
    try:
        cpp_loss, cpp_fp = _one_step(cpp_bm)
        cpp_draws = _continuation_draws()
    except Exception as error:
        out["cpp_vs_eager"] = {"error": f"{type(error).__name__}: {error}"}
        out["cpp_vs_eager_strict_equal"] = False
        out["cpp_normal_full_state_pass"] = False
        out["cpp_normal_full_state_diffs"] = ["cpp-step-failed"]
        out["cpp_loss_normal_tol_pass"] = False
        out["fp64_reference"] = {
            "source": STOCK_FP64_SOURCE,
            "built": False,
            "note": "not attempted: cpp step failed",
        }
        out["stock_reference_valid"] = None
        return out
    finally:
        _ = snap_cpp
    strict_eq, diffs = _fingerprints_equal(cpp_fp, ref_fp)
    # RNG continuation draws: strict includes bytewise draw equality plus the
    # full RNG-state sections (continuation draws prove the step did not
    # diverge the streams -- corpus/target randomness in both patterns).
    if _frozen_equal(cpp_draws, ref_draws):
        draws_ddiffs: list[str] = []
    else:
        draws_ddiffs = _diff_paths("rng_draws", cpp_draws, ref_draws)
    draws_eq = len(draws_ddiffs) == 0
    if not draws_eq:
        strict_eq = False
        diffs = diffs + draws_ddiffs[:20]
    out["cpp_vs_eager_strict_equal"] = bool(strict_eq)
    out["cpp_vs_eager_diffs"] = diffs[:100]
    out["cpp_vs_eager"] = {"bytewise_equal": bool(strict_eq), "diffs": diffs[:100]}
    out["rng_continuation_equal"] = bool(draws_eq)
    out["rng_continuation_diffs"] = draws_ddiffs[:20]
    # Normal lane: stock-effective full post-state acceptance (loss AND
    # grads, updated params, mutable buffers, optimizer state and
    # schedulers -- loss-only would miss wrong gradients/updates, and
    # buffers carry BatchNorm running state). Layout/ABI metadata fails
    # closed; values use the established stock comparator
    # (torch._dynamo.utils.same: allclose then FP64 RMSE 2x reference-error
    # envelope, cosine only per stock policy) with the genuine FP64
    # reference. RNG stays bytewise-strict (never allclose on RNG bytes).
    # Existing per-section assert_close-style checks remain as diagnostics.
    tb_name_t = model_name_of(eager_bm)
    try:
        tol_t, cosine_t, policy_t = _stock_tolerance_and_flags(tb_name_t, "train")
    except Exception as error:
        tol_t, cosine_t, policy_t = (
            1e-3,
            False,
            {
                "source": STOCK_TOLICY_SOURCE,
                "error": f"{type(error).__name__}: {error}",
            },
        )
    out["normal_policy"] = policy_t
    out["normal_comparator"] = {"source": STOCK_COMPARATOR_SOURCE}
    fp64_train_fp, fp64_train_record = _fp64_train_post_state(
        eager_bm, snap_clean, _one_step
    )
    out["fp64_reference"] = fp64_train_record
    eff_cosine_t = bool(cosine_t) or fp64_train_record.get("fallback") == "cosine"
    if eff_cosine_t and fp64_train_record.get("fallback") == "cosine":
        out["normal_policy"]["cosine_fallback"] = fp64_train_record.get(
            "fallback_reason"
        )
        out["normal_policy"]["cosine_fallback_source"] = fp64_train_record.get(
            "fallback_source"
        )
    out["normal_tolerance"] = {"tol": tol_t, "cosine": eff_cosine_t}
    out["normal_tolerances"] = {"rtol": NORMAL_RTOL, "atol": NORMAL_ATOL}
    normal_sections = (
        "loss",
        "grads",
        "params",
        "buffers",
        "optimizer_state",
        "schedulers",
    )
    normal_diffs: list[str] = []
    normal_ok = True
    for section in normal_sections:
        ref_s = ref_fp.get(section, "__missing__")
        cmp_s = cpp_fp.get(section, "__missing__")
        fp64_s = (
            fp64_train_fp.get(section, "__missing__")
            if isinstance(fp64_train_fp, dict)
            else "__missing__"
        )
        if ref_s == "__missing__" or cmp_s == "__missing__":
            if ref_s != cmp_s:
                normal_ok = False
                normal_diffs.append(f"{section}.missing")
            continue
        if fp64_s == "__missing__":
            fp64_arg: Any | None = None
        else:
            fp64_arg = fp64_s
        ok, reason = _stock_section_pass(
            ref_s, cmp_s, fp64_arg, tol=tol_t, cos_similarity=eff_cosine_t
        )
        if not ok:
            normal_ok = False
            normal_diffs.append(f"{section}: {reason}"[:300])
    out["cpp_normal_full_state_pass"] = bool(normal_ok)
    out["cpp_normal_full_state_diffs"] = normal_diffs[:100]
    # Normal-lane RNG: quantitative closeness on continuation draws only
    # (never assert_close across unrelated stochastic masks -- draws are
    # matched by construction via restoration; divergence is reported).
    rng_sections = ("torch_rng", "cuda_rng", "python_rng", "numpy_rng")
    rng_ok = True
    rng_diffs: list[str] = []
    for section in rng_sections:
        ref_s = ref_fp.get(section, "__missing__")
        cmp_s = cpp_fp.get(section, "__missing__")
        if ref_s == "__missing__" or cmp_s == "__missing__":
            if ref_s != cmp_s:
                rng_ok = False
                rng_diffs.append(f"{section}.missing")
            continue
        ok, sdiffs = _normal_section_close(
            ref_s, cmp_s, rtol=NORMAL_RTOL, atol=NORMAL_ATOL
        )
        if not ok:
            rng_ok = False
            rng_diffs.extend(f"{section}.{d}" for d in sdiffs[:10])
    out["cpp_rng_normal_close"] = bool(rng_ok)
    out["cpp_rng_normal_diffs"] = rng_diffs[:40]
    # Legacy loss-only key kept for continuity (subsumed by the full gate).
    out["cpp_loss_normal_tol_pass"] = (
        bool(normal_ok) if isinstance(cpp_loss, dict) or cpp_loss is None else None
    )
    try:
        if (
            isinstance(cpp_loss, dict)
            and cpp_loss.get("__tensor__") is True
            and isinstance(ref_loss, dict)
            and ref_loss.get("__tensor__") is True
        ):
            ok, reason = _normal_tensor_close(
                ref_loss, cpp_loss, rtol=NORMAL_RTOL, atol=NORMAL_ATOL
            )
            out["cpp_loss_normal_tol_pass"] = bool(ok)
            if not ok:
                out["cpp_loss_normal_tol_error"] = reason[:1000]
        elif cpp_loss is None and ref_loss is None:
            out["cpp_loss_normal_tol_pass"] = None
            out["cpp_loss_normal_tol_note"] = "loss unavailable; not fabricated"
        elif "cpp_loss_normal_tol_pass" not in out:
            out["cpp_loss_normal_tol_pass"] = None
    except Exception as error:
        out["cpp_loss_normal_tol_pass"] = False
        out["cpp_loss_normal_tol_error"] = (
            f"{type(error).__name__}: {str(error)[:1000]}"
        )
    if stock_bm is not None:
        try:
            restore_state(stock_bm, snap_clean)
            stock_loss, stock_fp = _one_step(stock_bm)
            seq, sdiffs = _fingerprints_equal(stock_fp, ref_fp)
            out["stock_vs_eager_strict_equal"] = bool(seq)
            out["stock_vs_eager_diffs"] = sdiffs[:50]
            # Stock vs same eager + FP64 reference via the stock comparator.
            stock_sections_ok = True
            stock_sections_diffs: list[str] = []
            for section in normal_sections:
                ref_s = ref_fp.get(section, "__missing__")
                stk_s = stock_fp.get(section, "__missing__")
                fp64_s = (
                    fp64_train_fp.get(section, "__missing__")
                    if isinstance(fp64_train_fp, dict)
                    else "__missing__"
                )
                if ref_s == "__missing__" or stk_s == "__missing__":
                    if ref_s != stk_s:
                        stock_sections_ok = False
                        stock_sections_diffs.append(f"{section}.missing")
                    continue
                ok, reason = _stock_section_pass(
                    ref_s,
                    stk_s,
                    None if fp64_s == "__missing__" else fp64_s,
                    tol=tol_t,
                    cos_similarity=eff_cosine_t,
                )
                if not ok:
                    stock_sections_ok = False
                    stock_sections_diffs.append(f"{section}: {reason}"[:300])
            out["stock_vs_eager_stock_pass"] = bool(stock_sections_ok)
            out["stock_vs_eager_stock_diffs"] = stock_sections_diffs[:50]
            # CPP-vs-stock diagnostic (never acceptance).
            cpp_stock_ok = True
            cpp_stock_diffs: list[str] = []
            for section in normal_sections:
                stk_s = stock_fp.get(section, "__missing__")
                cmp_s = cpp_fp.get(section, "__missing__")
                fp64_s = (
                    fp64_train_fp.get(section, "__missing__")
                    if isinstance(fp64_train_fp, dict)
                    else "__missing__"
                )
                if stk_s == "__missing__" or cmp_s == "__missing__":
                    if stk_s != cmp_s:
                        cpp_stock_ok = False
                        cpp_stock_diffs.append(f"{section}.missing")
                    continue
                ok, reason = _stock_section_pass(
                    stk_s,
                    cmp_s,
                    None if fp64_s == "__missing__" else fp64_s,
                    tol=tol_t,
                    cos_similarity=eff_cosine_t,
                )
                if not ok:
                    cpp_stock_ok = False
                    cpp_stock_diffs.append(f"{section}: {reason}"[:300])
            out["cpp_vs_stock"] = {
                "stock_pass": bool(cpp_stock_ok),
                "diffs": cpp_stock_diffs[:50],
            }
            _ = stock_loss
        except RestorationError as error:
            out["stock_vs_eager"] = {
                "error_type": "RestorationError",
                "error": f"stock restore: {error}",
            }
            out["stock_vs_eager_stock_pass"] = False
        except Exception as error:
            out["stock_vs_eager"] = {"error": f"{type(error).__name__}: {error}"}
            out["stock_vs_eager_stock_pass"] = False
    else:
        out["stock_vs_eager_stock_pass"] = None
    stock_ok_t = out.get("stock_vs_eager_stock_pass")
    out["stock_reference_valid"] = stock_ok_t if stock_ok_t is not None else None
    return out


# ---------------------------------------------------------------------------
# Timing: vetted interleave, serialized, never for failed runs.
# ---------------------------------------------------------------------------


def timed_latencies(
    get_functions: Any, warmup: int, repeats: int
) -> dict[str, dict[str, Any]]:
    """Vetted interleaved GPU timing over zero-arg callables.

    ``get_functions`` maps label -> thunk (a fresh zero-arg callable per
    label, rebuilt by the caller after a full state restore so every
    variant starts from the identical initial state). Warmup is
    round-robin across variants (never eager-N then cpp-N, which would
    consume global RNG asymmetrically); measurement rotates the variant
    order each repeat. Serialized GPU timing with the vetted benchmarker;
    never CPU host timing.
    """
    import torch
    from torch._inductor.runtime.benchmarking import benchmarker

    labels = list(get_functions)
    for _ in range(warmup):
        for label in labels:
            get_functions[label]()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    samples: dict[str, list[float]] = {label: [] for label in labels}
    for repeat in range(repeats):
        order = labels[repeat % len(labels) :] + labels[: repeat % len(labels)]
        for label in order:
            ms = benchmarker.benchmark_gpu(
                get_functions[label], is_vetted_benchmarking=True
            )
            samples[label].append(float(ms))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    result: dict[str, dict[str, Any]] = {}
    for label, values in samples.items():
        result[label] = {
            "samples_ms": values,
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
        }
    return result


# ---------------------------------------------------------------------------
# Per-case orchestration (partial failures never abort the suite).
# ---------------------------------------------------------------------------


def run_case(
    model_name: str,
    test: str,
    mode: str,
    batch_size: int,
    warmup: int,
    repeats: int,
    accuracy_only: bool,
    load_only: bool,
    profile: bool,
    artifacts_root: Path,
    settings_before: dict[str, Any],
) -> dict[str, Any]:
    import torch

    case: dict[str, Any] = {
        "model": model_name,
        "test": test,
        "cpp_numerics": mode,
        "batch_size": batch_size,
        "extra_args": list(EXTRA_ARGS),
        "device": DEVICE,
        "errors": [],
        "status": "pending",
    }
    if not torch.cuda.is_available():
        case.update(
            {"status": "skip", "reason": "CUDA is required; skip is never a pass"}
        )
        return case
    # -- inventory (genuine factory; records incoming-globals effects) ------
    try:
        inv_bm = construct_benchmark(model_name, test, batch_size)
    except Exception as error:
        case.update(
            {
                "status": "fail",
                "error_type": type(error).__name__,
                "error": str(error)[:2000],
            }
        )
        case["errors"].append(f"construction failed: {error}")
        return case
    case["inventory"] = inventory_benchmark(inv_bm, model_name, test, batch_size)
    case["settings_before_model_construction"] = settings_before
    try:
        case["settings_after_model_construction"] = numerical_settings()
    except Exception:
        case["settings_after_model_construction"] = {}
    if case["settings_after_model_construction"] != settings_before:
        case["model_loading_changed_globals"] = True
    else:
        case["model_loading_changed_globals"] = False
    try:
        del inv_bm
        torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass
    if load_only:
        case["status"] = "inventory-only"
        case["note"] = "load-only: no compilation, no accuracy, NO latency measurement"
        return case

    eager_bm = cpp_bm = stock_bm = None
    try:
        eager_bm = construct_benchmark(model_name, test, batch_size)
        snap_clean = snapshot_state(eager_bm)
    except Exception as error:
        case.update(
            {
                "status": "fail",
                "error_type": type(error).__name__,
                "error": str(error)[:2000],
            }
        )
        return case
    # -- CPP compile --------------------------------------------------------
    cpp_info: dict[str, Any] = {}
    compile_cleanup_failed: dict[str, Any] | None = None
    try:
        cpp_bm = construct_benchmark(model_name, test, batch_size)
        snap_cpp_clean = snapshot_state(cpp_bm)
        cpp_info = compile_variant(cpp_bm, mode, test)
        # Restore post-compile mutation to the clean start state. A cleanup
        # failure vetoes accuracy AND timing even when compilation itself
        # succeeded: the variant's start state is unproven, so no execution
        # validation may run on it (finding 4).
        try:
            restore_state(cpp_bm, snap_cpp_clean)
        except RestorationError as error:
            compile_cleanup_failed = {
                "error_type": "RestorationError",
                "error": f"compile_cleanup cpp restore: {error}",
                "stage": "compile_cleanup",
            }
        except Exception as error:
            compile_cleanup_failed = {
                "error_type": type(error).__name__,
                "error": f"compile_cleanup cpp restore: {error}",
                "stage": "compile_cleanup",
            }
        case["cpp_compile"] = {
            k: v
            for k, v in cpp_info.items()
            if k not in ("triton_sources_available", "artifact_texts")
        }
        case["cpp_coverage"] = cpp_info.get("coverage", {})
        case["cpp_graphs"] = {
            "num_seam_graphs": cpp_info.get("num_seam_graphs"),
            "num_seam_graphs_ok": cpp_info.get("num_seam_graphs_ok"),
            "forward_cpp_evidence": cpp_info.get("forward_cpp_evidence"),
            "backward_cpp_evidence": cpp_info.get("backward_cpp_evidence"),
            "backend_graph_kind": cpp_info.get("backend_graph_kind"),
            "collector_retains_only_last_graph": True,
            "calls": cpp_info.get("seam_calls", []),
        }
        case["graph_direction_coverage"] = {
            "forward_cpp_graphs_ok": bool(cpp_info.get("forward_cpp_evidence")),
            "backward_cpp_graphs_ok": bool(cpp_info.get("backward_cpp_evidence")),
            "num_seam_graphs_ok": cpp_info.get("num_seam_graphs_ok"),
        }
        # Persist per-graph sources/IR to disk (scorecard keeps paths only;
        # large blobs never enter report.json).
        try:
            case_dir = artifacts_root / f"{model_name}_{test}_{mode}"
            case_dir.mkdir(parents=True, exist_ok=True)
            case["artifact_dir"] = str(case_dir)
            texts = cpp_info.get("artifact_texts", {}) or {}
            written: list[str] = []
            for key in (
                "program_dump",
                "optimized_program_dump",
                "kernel_dump",
                "tiled_dump",
            ):
                text = texts.get(key)
                if isinstance(text, str) and text:
                    (case_dir / f"{key}.txt").write_text(text)
                    written.append(f"{key}.txt")
            for fname, src in (texts.get("triton_sources", {}) or {}).items():
                safe = "".join(
                    c if (c.isalnum() or c in "-_.") else "_" for c in str(fname)
                )
                (case_dir / f"triton_{safe}.py").write_text(src)
                written.append(f"triton_{safe}.py")
            for fname, ptx in (texts.get("triton_ptx", {}) or {}).items():
                safe = "".join(
                    c if (c.isalnum() or c in "-_.") else "_" for c in str(fname)
                )
                (case_dir / f"ptx_{safe}.ptx").write_text(ptx)
                written.append(f"ptx_{safe}.ptx")
            try:
                (case_dir / "coverage.json").write_text(
                    json.dumps(
                        {
                            "coverage": cpp_info.get("coverage", {}),
                            "function_names": list(
                                (cpp_info.get("coverage", {}) or {}).get(
                                    "function_names", []
                                )
                            ),
                            "seam_calls": cpp_info.get("seam_calls", []),
                        },
                        indent=2,
                        sort_keys=True,
                        default=str,
                    )
                )
                written.append("coverage.json")
            except Exception:
                pass
            case["artifacts"] = sorted(written)
            case["source_paths"] = sorted(str(case_dir / name) for name in written)
        except Exception as error:
            case["errors"].append(f"artifact persistence failed: {error}")
            try:
                case_dir = artifacts_root / f"{model_name}_{test}_{mode}"
                case_dir.mkdir(parents=True, exist_ok=True)
                case["artifact_dir"] = str(case_dir)
            except Exception:
                pass
        try:
            after = numerical_settings()
            case["settings_after_cpp_compile"] = after
            case["compiler_changed_globals"] = (
                after != case["settings_after_model_construction"]
            )
        except Exception:
            case["compiler_changed_globals"] = None
    except Exception as error:
        case["cpp_compile"] = {
            "error_type": type(error).__name__,
            "error": str(error)[:2000],
        }
        case["errors"].append(f"cpp compile failed: {type(error).__name__}: {error}")
        # Preserve per-graph rejection evidence carried by the failure (e.g.
        # the scoped BatchNorm boundary error after a seam attempt). This is
        # diagnostic data, never a pass.
        partial_calls = getattr(error, "_loops_ir_seam_calls", None)
        if isinstance(partial_calls, list) and partial_calls:
            try:
                summary = _summarize_seam_calls(partial_calls)
                case["cpp_graphs"] = summary
                case["graph_direction_coverage"] = {
                    "forward_cpp_graphs_ok": summary["forward_cpp_evidence"],
                    "backward_cpp_graphs_ok": summary["backward_cpp_evidence"],
                    "num_seam_graphs_ok": summary["num_seam_graphs_ok"],
                }
            except Exception:
                pass
    # -- stock compile (reference) -------------------------------------------
    stock_info: dict[str, Any] = {}
    stock_cleanup_failed: dict[str, Any] | None = None
    try:
        stock_bm = construct_benchmark(model_name, test, batch_size)
        snap_stock_clean = snapshot_state(stock_bm)
        stock_info = compile_stock_variant(stock_bm, test)
        try:
            restore_state(stock_bm, snap_stock_clean)
        except RestorationError as error:
            stock_cleanup_failed = {
                "error_type": "RestorationError",
                "error": f"compile_cleanup stock restore: {error}",
                "stage": "compile_cleanup",
            }
        except Exception as error:
            stock_cleanup_failed = {
                "error_type": type(error).__name__,
                "error": f"compile_cleanup stock restore: {error}",
                "stage": "compile_cleanup",
            }
        case["stock_compile"] = stock_info
    except Exception as error:
        case["stock_compile"] = {
            "error_type": type(error).__name__,
            "error": str(error)[:2000],
        }
        case["errors"].append(f"stock compile failed: {type(error).__name__}: {error}")
        stock_bm = None
    # Compile-cleanup veto (finding 4): a variant whose post-compile start
    # state is unproven must not run execution validation. The compile itself
    # may be recorded as OK (evidence preserved), but the variant is None'd
    # so accuracy compares nothing against itself, and timing is vetoed.
    if compile_cleanup_failed is not None:
        case["errors"].append(
            f"cpp {compile_cleanup_failed['error_type']}: "
            f"{compile_cleanup_failed['error']}"
        )
        case["cpp_cleanup_veto"] = compile_cleanup_failed
        cpp_bm = None
    if stock_cleanup_failed is not None:
        case["errors"].append(
            f"stock {stock_cleanup_failed['error_type']}: "
            f"{stock_cleanup_failed['error']}"
        )
        case["stock_cleanup_veto"] = stock_cleanup_failed
        stock_bm = None
    # -- accuracy -------------------------------------------------------------
    # A failed CPP compile must NEVER compare a variant against itself:
    # cpp_bm_for_acc is None on failure (verdict fail via compile error).
    cpp_bm_for_acc = (
        cpp_bm
        if cpp_bm is not None
        and "error_type" not in (case.get("cpp_compile", {}) or {})
        and "error" not in (case.get("cpp_compile", {}) or {})
        else None
    )
    try:
        try:
            restore_state(eager_bm, snap_clean)
        except RestorationError as error:
            raise RestorationError(f"accuracy: eager restore: {error}") from error
        if cpp_bm_for_acc is not None:
            try:
                restore_state(cpp_bm_for_acc, snap_clean)
            except RestorationError as error:
                raise RestorationError(f"accuracy: cpp restore: {error}") from error
        if stock_bm is not None:
            try:
                restore_state(stock_bm, snap_clean)
            except RestorationError as error:
                raise RestorationError(f"accuracy: stock restore: {error}") from error
        if test == "eval":
            acc = eval_accuracy(eager_bm, cpp_bm_for_acc, stock_bm)
        else:
            acc = train_accuracy(eager_bm, cpp_bm_for_acc, stock_bm)
        case["accuracy"] = acc
    except RestorationError as error:
        # Restore failure vetoes accuracy AND timing: the case is invalid,
        # explicitly recorded, never timed, never a pass.
        case["accuracy"] = {
            "error_type": "RestorationError",
            "error": str(error)[:2000],
        }
        case["errors"].append(f"state restore failed: {error}")
        case["status"] = "fail"
        case["timing"] = {"skipped": f"not timed: {error}"}
        return case
    except Exception as error:
        case["accuracy"] = {
            "error_type": type(error).__name__,
            "error": str(error)[:2000],
        }
        case["errors"].append(f"accuracy failed: {type(error).__name__}: {error}")
    cpp_bm = cpp_bm_for_acc if cpp_bm_for_acc is not None else None
    # -- verdict ----------------------------------------------------------------
    case["status"] = _verdict(case, test, mode)
    # -- timing (never for failed runs; skipped under --accuracy-only) ---------
    if accuracy_only:
        case["timing"] = {"skipped": "accuracy-only"}
    elif case["status"] != "pass":
        case["timing"] = {"skipped": "not timed: failed run is never valid"}
    else:
        try:
            # Every variant is restored from the identical clean snapshot
            # immediately before warmup, so post-accuracy state drift can
            # never leak into timing. Each training invoke runs a whole
            # optimization step from the same initial state; each variant
            # gets identical RNG treatment (restore then warmup advances
            # each variant's own stream symmetrically -- never eager-2 /
            # cpp-4 / stock-6 global-RNG consumption). Eval runs under
            # inference mode with strict stable states.
            import torch as _torch

            def _restore_all() -> None:
                try:
                    restore_state(eager_bm, snap_clean)
                except RestorationError as error:
                    raise RestorationError(f"timing: eager restore: {error}") from error
                if cpp_bm is not None:
                    try:
                        restore_state(cpp_bm, snap_clean)
                    except RestorationError as error:
                        raise RestorationError(
                            f"timing: cpp restore: {error}"
                        ) from error
                if stock_bm is not None:
                    try:
                        restore_state(stock_bm, snap_clean)
                    except RestorationError as error:
                        raise RestorationError(
                            f"timing: stock restore: {error}"
                        ) from error

            _restore_all()
            get_functions: dict[str, Any] = {}
            if test == "eval":
                get_functions["eager"] = lambda: eager_bm.invoke()
                get_functions[f"cpp_{mode}"] = lambda: cpp_bm.invoke()
                if stock_bm is not None:
                    get_functions["stock"] = lambda: stock_bm.invoke()
                with _torch.inference_mode():
                    lat = timed_latencies(get_functions, warmup, repeats)
            else:
                get_functions["eager"] = lambda: eager_bm.invoke()
                get_functions[f"cpp_{mode}"] = lambda: cpp_bm.invoke()
                if stock_bm is not None:
                    get_functions["stock"] = lambda: stock_bm.invoke()
                lat = timed_latencies(get_functions, warmup, repeats)
            case["timing"] = lat
            try:
                eager_med = lat["eager"]["median_ms"]
                for label, vals in lat.items():
                    if label != "eager":
                        vals["speedup_over_eager"] = (
                            eager_med / vals["median_ms"] if vals["median_ms"] else None
                        )
                if "stock" in lat and f"cpp_{mode}" in lat:
                    case["timing"][f"cpp_{mode}"]["raw_ratio_vs_stock"] = (
                        lat["stock"]["median_ms"] / lat[f"cpp_{mode}"]["median_ms"]
                        if lat[f"cpp_{mode}"]["median_ms"]
                        else None
                    )
            except Exception:
                pass
        except RestorationError as error:
            case["timing"] = {"error": f"RestorationError: {error}"}
            case["errors"].append(f"timing restore failed: {error}")
            if case["status"] == "pass":
                case["status"] = "fail"
        except Exception as error:
            case["timing"] = {"error": f"{type(error).__name__}: {error}"}
            case["errors"].append(f"timing failed: {error}")
            if case["status"] == "pass":
                case["status"] = "fail"
    if profile:
        case["profile"] = {
            "enabled": True,
            "compile_seconds": (case.get("cpp_compile", {}) or {}).get(
                "compile_and_first_run_seconds"
            ),
        }
    for bm in (eager_bm, cpp_bm, stock_bm):
        try:
            del bm
        except Exception:
            pass
    try:
        torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass
    return case


def _verdict(case: dict[str, Any], test: str, mode: str) -> str:
    cc = case.get("cpp_compile", {})
    if isinstance(cc, dict) and ("error" in cc or "error_type" in cc):
        return "fail"
    cov = case.get("cpp_coverage", {}) or {}
    try:
        if int(cov.get("num_generated_kernels", 0)) <= 0:
            return "fail"
    except Exception:
        return "fail"
    acc = case.get("accuracy", {}) or {}
    if isinstance(acc, dict) and ("error" in acc or "error_type" in acc):
        return "fail"
    if test == "eval":
        if mode == "strict_eager":
            if acc.get("eager_repeat_bytewise") is not True:
                return "fail"
            cmpd = acc.get("cpp_vs_eager", {}) or {}
            if cmpd.get("bytewise_equal") is not True:
                return "fail"
            return "pass"
        else:
            # Normal acceptance: established stock-effective comparator
            # (same eager + FP64 reference). Legacy assert_close fields stay
            # as diagnostics only. A real valid stock reference is required:
            # never mask a stock failure or claim performance on an
            # unproven reference.
            if acc.get("stock_reference_valid") is False:
                return "fail"
            if acc.get("cpp_vs_eager_stock_pass") is True:
                return "pass"
            return "fail"
    # train: both modes require compiled-backward evidence (forward-only
    # success is an explicit failure); strict additionally requires eager
    # stability + bytewise post-state, normal requires full post-state
    # closeness (loss AND grads/params/optimizer/schedulers) so a wrong
    # gradient or optimizer update cannot hide behind a close loss.
    graphs = case.get("cpp_graphs", {}) or {}
    if graphs.get("backward_cpp_evidence") is not True:
        return "fail"
    if mode == "strict_eager":
        if acc.get("eager_repeat_stable") is not True:
            return "fail"
        if acc.get("cpp_vs_eager_strict_equal") is not True:
            return "fail"
        return "pass"
    else:
        # Normal acceptance: stock-effective full-state gate (loss AND
        # grads/params/buffers/optimizer/schedulers via layout + same with
        # the FP64 reference). Legacy assert_close-style sections stay as
        # diagnostics. Stock must be a real valid reference.
        if acc.get("stock_reference_valid") is False:
            return "fail"
        if (
            acc.get("cpp_normal_full_state_pass") is True
            and graphs.get("num_seam_graphs_ok", 0) >= 1
        ):
            return "pass"
        return "fail"


def _completion_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    passes = sum(case.get("status") == "pass" for case in cases)
    fails = sum(case.get("status") == "fail" for case in cases)
    selected_complete = bool(cases) and passes == len(cases)
    required = {
        (model, test, mode)
        for model in MODEL_MANIFEST
        for test in ("eval", "train")
        for mode in ("strict_eager", "inductor")
    }
    passed_default_cases = {
        (case["model"], case["test"], case.get("cpp_numerics"))
        for case in cases
        if case.get("status") == "pass"
        and case.get("model") in MODEL_MANIFEST
        and case.get("test") in ("eval", "train")
        and case.get("batch_size")
        == MODEL_MANIFEST[case["model"]][f"{case['test']}_bs"]
    }
    return {
        "total": len(cases),
        "pass": passes,
        "fail": fails,
        "skip_or_inventory": len(cases) - passes - fails,
        "selected_cases_complete": selected_complete,
        "complete": selected_complete and required <= passed_default_cases,
        "note": (
            "Completion requires all 40 CPP cases passing at manifest batch sizes "
            "(10 models x eval/train x strict/inductor) with generated-code "
            "participation and backward-CPP evidence for training. Partial "
            "progress is explicitly incomplete; skips are never passes. "
            "The verdict and exit code describe the selected cases only."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(tempfile.mkdtemp(prefix="loops_ir_torchbench_suite_"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_root = out_dir / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    print(f"output_dir={out_dir}", flush=True)

    try:
        settings_before_any = numerical_settings()
    except Exception:
        settings_before_any = {}
    report: dict[str, Any] = {
        "suite": "loops_ir_torchbench_suite",
        "harness": str(Path(__file__).resolve()),
        "models": list(args.models),
        "tests": tests_for_arg(args.test),
        "cpp_numerics": modes_for_arg(args.cpp_numerics),
        "batch_manifest": {m: dict(MODEL_MANIFEST[m]) for m in args.models},
        "extra_args": list(EXTRA_ARGS),
        "device": DEVICE,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "device_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "profile": bool(args.profile),
        "accuracy_only": bool(args.accuracy_only),
        "load_only": bool(args.load_only),
        "normal_tolerances": {"rtol": NORMAL_RTOL, "atol": NORMAL_ATOL},
        "normal_policy_sources": {
            "tolerance": STOCK_TOLICY_SOURCE,
            "check": STOCK_CHECK_SOURCE,
            "comparator": STOCK_COMPARATOR_SOURCE,
            "fp64_reference": STOCK_FP64_SOURCE,
        },
        "settings_before_any_construction": settings_before_any,
        "cases": [],
        "errors": [],
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))

    total = len(args.models) * len(report["tests"]) * len(report["cpp_numerics"])
    print(
        f"cases={total} (models={len(args.models)} x tests={len(report['tests'])} x numerics={len(report['cpp_numerics'])})",
        flush=True,
    )
    for model_name in args.models:
        for test in report["tests"]:
            manifest = MODEL_MANIFEST[model_name]
            batch_size = (
                manifest["train_bs"] if test == "train" else manifest["eval_bs"]
            )
            for mode in report["cpp_numerics"]:
                print(
                    f"--- case {model_name}/{test}/{mode} bs={batch_size} ---",
                    flush=True,
                )
                try:
                    case = run_case(
                        model_name,
                        test,
                        mode,
                        batch_size,
                        args.warmup,
                        args.repeats,
                        bool(args.accuracy_only),
                        bool(args.load_only),
                        bool(args.profile),
                        artifacts_root,
                        dict(settings_before_any),
                    )
                except Exception as error:
                    case = {
                        "model": model_name,
                        "test": test,
                        "cpp_numerics": mode,
                        "status": "fail",
                        "error_type": type(error).__name__,
                        "error": str(error)[:2000],
                        "errors": [str(error)[:2000]],
                    }
                report["cases"].append(case)
                # Incremental scorecard (never truncate on later failures).
                try:
                    report_path.write_text(
                        json.dumps(report, indent=2, sort_keys=True, default=str)
                    )
                except Exception as error:
                    print(f"scorecard write failed: {error}", flush=True)
                print(
                    f"case {model_name}/{test}/{mode}: {case.get('status')}", flush=True
                )

    report["summary"] = _completion_summary(report["cases"])
    report["verdict"] = (
        "PASS" if report["summary"]["selected_cases_complete"] else "FAIL"
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"output_dir={out_dir}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

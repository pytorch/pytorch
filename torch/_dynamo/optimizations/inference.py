import base64
import hashlib
import io
import itertools
import json
import logging
import os
import time
from collections import defaultdict

import torch

from .. import config
from ..utils import (
    check_is_cuda,
    checkpoint_params,
    clone_inputs,
    count_calls,
    counters,
)
from .normalize import long_name, normalize_ir

log = logging.getLogger(__name__)


def string_key(gm: torch.fx.GraphModule, example_inputs):
    out = io.StringIO()
    node_to_id = defaultdict(iter(itertools.count()).__next__)

    def argkey(n: torch.fx.Node):
        return f"#{node_to_id[n]}"

    def tensorkey(t):
        if isinstance(t, torch.Tensor):
            requires_grad = t.requires_grad and torch.torch.is_grad_enabled()
            return (
                f"{t.__class__.__name__}({t.dtype}, {t.device}, "
                f"{tuple(t.size())}, {tuple(t.stride())}, {requires_grad})"
            )
        return type(t).__name__

    inputs_iter = iter(example_inputs)

    for node in gm.graph.nodes:
        key = argkey(node)
        name = "."
        if node.op == "placeholder":
            name = tensorkey(next(inputs_iter))
        elif node.op == "get_attr":
            val = eval(f"self.{node.target}", {"self": gm})
            name = tensorkey(val)
        elif node.op in ("call_function", "call_method", "call_module"):
            name = long_name(gm, node)
        out.write(
            f"{key} {node.op} {name} "
            f"{torch.fx.map_arg(node.args, argkey)!r} "
            f"{torch.fx.map_arg(node.kwargs, argkey)!r}\n"
        )
    return out.getvalue()


def graph_hash(gm: torch.fx.GraphModule, example_inputs):
    return "g" + base64.urlsafe_b64encode(
        hashlib.sha256(string_key(gm, example_inputs).encode("utf-8")).digest()
    )[:39].decode("utf-8")


def folder_name(gm: torch.fx.GraphModule, example_inputs):
    base = os.path.join(config.base_dir, "subgraphs")
    if not os.path.exists(base):
        os.mkdir(base)
        open(os.path.join(base, "__init__.py"), "w").close()
    return os.path.join(base, graph_hash(gm, example_inputs))


def record_graph_stats(gm):
    for node in gm.graph.nodes:
        if node.op in ("call_function", "call_method", "call_module"):
            counters[node.op][long_name(gm, node)] += 1
        elif node.op in ("placeholder", "output", "get_attr"):
            pass
        else:
            raise AssertionError(node.op)


def check_requires_grad(gm, example_inputs):
    if torch.is_grad_enabled():
        if any(
            getattr(x, "requires_grad", False)
            for x in itertools.chain(example_inputs, gm.parameters(True))
        ):
            return True
    return False


def jit_trace(gm, example_inputs):
    """Wrapper around jit.trace to handle hooks"""
    restore_backward_hooks = []

    def visit(mod):
        if mod._backward_hooks:
            restore_backward_hooks.append((mod, mod._backward_hooks))
            mod._backward_hooks = []

    if not check_requires_grad(gm, example_inputs):
        # in inference mode it is safe to ignore backwards hooks to allow tracing
        gm.apply(visit)

    try:
        return torch.jit.trace(gm.forward, example_inputs)
    finally:
        for mod, hooks in restore_backward_hooks:
            mod._backward_hooks = hooks


def same(left, right):
    return len(left) == len(right) and all(
        torch.allclose(a, b, atol=1e-4, rtol=1e-4) for a, b in zip(left, right)
    )


class TorchScriptStrategy(object):
    """Common base for backend strategies that use TorchScript"""

    @classmethod
    def compile_fn(cls, gm: torch.fx.GraphModule, example_inputs):
        if count_calls(gm.graph) < 2:
            return gm.forward  # no point for tiny graphs
        return cls(gm, example_inputs).verified_candidate()

    def __init__(self, gm: torch.fx.GraphModule, example_inputs):
        super(TorchScriptStrategy, self).__init__()
        self.restore = checkpoint_params(gm)
        self.original_example_inputs = example_inputs
        self.correct = gm.forward(*self.example_inputs)
        self.gm = normalize_ir(gm, self.original_example_inputs)
        self.scripted = jit_trace(self.gm, self.example_inputs)

    @property
    def example_inputs(self):
        return clone_inputs(self.original_example_inputs)

    def verified_candidate(self):
        try:
            candidate = self.candidate()
            if candidate is None or candidate is self.gm.forward:
                return self.gm.forward

            self.restore()
            result = candidate(*self.example_inputs)

            if same(result, self.correct):
                return candidate

            print(f"incorrect candidate {self}")

            return self.gm.forward
        except Exception:
            log.exception("error in verified_candidate()")
            return self.gm.forward
        finally:
            self.restore()

    def candidate(self):
        raise NotImplementedError()


def save_pt(path, name, data):
    with open(os.path.join(path, name), "wb") as fd:
        torch.save(data, fd)


def save_metadata(path, gm, example_inputs):
    with open(os.path.join(path, "metadata.json"), "w") as fd:
        json.dump(
            {
                "is_cuda": check_is_cuda(gm, example_inputs),
            },
            fd,
        )


def touch_timestamp(path):
    open(os.path.join(path, "timestamp"), "w").write(str(time.time()))


def argmin(perf):
    best = "eager"
    best_sec = float("inf")
    for name, sec in perf.items():
        if sec < best_sec:
            best = name
            best_sec = float(sec)
            if name == "eager":
                # small bias torwards using eager since it is more robust
                best_sec *= 0.99
    return best

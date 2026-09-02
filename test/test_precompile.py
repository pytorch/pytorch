# Owner(s): ["oncall: pt2"]
import ast
import base64
import copy
import enum
import io
import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from unittest import mock

import sympy

import torch
import torch.utils._pytree as _pytree
from torch._dynamo.decorators import mark_dynamic, mark_unbacked
from torch._precompile import _dynamo_backend_source_literal, PrecompileError
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfCrossRef,
    skipIfTorchDynamo,
    TEST_NUMPY,
    TestCase,
)


# A module-level (global) model + a function referencing it, to exercise the
# constant-tensor guard against a baked global.
_GLOBAL_TENSOR = torch.randn(3)
_DYNAMO_INPUT_GLOBAL = torch.randn(3)
_DYNAMO_TENSOR_DEFAULT = torch.randn(3)


def _precompile_dynamo_dynamic(x):
    return x.sin() + x.shape[0]


def _precompile_dynamo_torch_sin(x):
    return torch.sin(x)


def _precompile_dynamo_pytree_sum(x):
    d = {"a": x, "b": x + 1}
    return sum(torch.utils._pytree.tree_leaves(d))


def _precompile_dynamo_sin_and_passthrough(x):
    return torch.sin(x), x


def _precompile_dynamo_input_global_identity(x):
    return x + 1 if x is _DYNAMO_INPUT_GLOBAL else x - 1


def _precompile_dynamo_tensor_default(x, bias=_DYNAMO_TENSOR_DEFAULT):
    return x + bias


def _precompile_dynamo_varargs(*xs):
    return xs[0] + xs[1]


def _precompile_dynamo_independent_outputs(x, y):
    return x.sin(), y.cos()


def _precompile_dynamo_nondiff_second_output(x):
    return x.sin(), x.detach() + 1


def _precompile_dynamo_inplace_add(x, y):
    x.add_(y)
    return x.sum()


def _precompile_dynamo_inplace_copy(x, src):
    x.copy_(src * 2)
    return x + 1


def _precompile_dynamo_dynamic_branch(x):
    if x.shape[0] == 1:
        return x + 100
    return x + 1


def _precompile_dynamo_scalar(x, scale):
    return x + scale


def _precompile_dynamo_aliasing(a, b):
    a.add_(1)
    return a * b


def _precompile_dynamo_dict_order(x, values):
    for value in values.values():
        x = x * value + 1
    return x


def _precompile_dynamo_graph_break(x):
    y = x + 1
    torch._dynamo.graph_break()
    return y * 2


def _precompile_dynamo_unpicklable_default(x, cb=lambda t: t):
    return x + 1


class _DynamoTensorHolder:
    def __init__(self):
        self.bias = torch.randn(3)


_DYNAMO_OBJECT_DEFAULT = _DynamoTensorHolder()


def _precompile_dynamo_object_tensor_default(x, cfg=_DYNAMO_OBJECT_DEFAULT):
    return x + 1


_DYNAMO_ENV_SCALE = 3


def _precompile_dynamo_env_scale(x):
    return x * _DYNAMO_ENV_SCALE


_DYNAMO_MUTATED_GLOBAL = None


def _precompile_dynamo_global_tensor(x):
    return x + _GLOBAL_TENSOR


def _precompile_dynamo_mutates_global(x):
    global _DYNAMO_MUTATED_GLOBAL
    _DYNAMO_MUTATED_GLOBAL = x
    return x + 1


def _precompile_dynamo_gather(x, idx):
    return x.sin()[idx] + x.shape[0]


def _precompile_dynamo_iter_sum(it, t):
    return t + next(it) + next(it)


def _precompile_dynamo_affine(x):
    return x * 2 + 1


def _precompile_dynamo_inplace_step(x):
    x.add_(1)
    return x * 2


def _precompile_dynamo_param_scale(p, x):
    return (p * x).sum()


def _precompile_dynamo_grad_step(x):
    if x.grad is not None:
        return x - 0.1 * x.grad
    return x.clone()


class _PrecompileDynamoTensorClassAttr:
    tensor = torch.randn(3)


def _precompile_dynamo_class_attr_tensor(x):
    return x + _PrecompileDynamoTensorClassAttr.tensor


_DYNAMO_TENSOR_MODULE = types.ModuleType("_precompile_dynamo_tensor_module")
_DYNAMO_TENSOR_MODULE.weight = torch.randn(3)


def _precompile_dynamo_module_attr_tensor(x):
    return x + _DYNAMO_TENSOR_MODULE.weight


class _PrecompileDynamoSlottedClassAttr:
    __slots__ = ()
    tensor = torch.randn(3)


_DYNAMO_SLOTTED_CLASS_ATTR = _PrecompileDynamoSlottedClassAttr()


def _precompile_dynamo_slotted_class_attr_tensor(x):
    return x + _DYNAMO_SLOTTED_CLASS_ATTR.tensor


class _PrecompileDynamoSlottedValue:
    __slots__ = ("t",)

    def __init__(self):
        self.t = torch.randn(3)


_DYNAMO_SLOTTED_VALUE = _PrecompileDynamoSlottedValue()


def _precompile_dynamo_slot_value_tensor(x):
    return x + _DYNAMO_SLOTTED_VALUE.t


class _PrecompileDynamoBoxWithModuleClassAttr:
    helper = torch.nn.Linear(2, 2)

    def __init__(self, scale):
        self.scale = scale


def _precompile_dynamo_box_scale(x, box):
    return x * box.scale


class _PrecompileDynamoAliasPayload:
    pass


class _PrecompileDynamoDtypeConfig:
    dtype = torch.float32
    memory_format = torch.channels_last
    layout = torch.strided


def _precompile_dynamo_dtype_branch(x, dtype):
    return x * 2 if dtype is _PrecompileDynamoDtypeConfig.dtype else x * 3


def _precompile_dynamo_format_branch(x, fmt):
    return x * 2 if fmt is _PrecompileDynamoDtypeConfig.memory_format else x * 3


def _precompile_dynamo_layout_branch(x, layout):
    return x * 2 if layout is _PrecompileDynamoDtypeConfig.layout else x * 3


class _PrecompileDynamoMode(enum.Enum):
    A = 1
    B = 2


def _precompile_dynamo_enum_passthrough(x, mode):
    return x * _PrecompileDynamoMode.A.value


def _precompile_dynamo_enum_branch(x, mode):
    return x * 2 if mode is _PrecompileDynamoMode.A else x * 3


class _PrecompileDynamoAliasHolder:
    payload = None


def _precompile_dynamo_class_attr_alias(x, p):
    return x * 2 if _PrecompileDynamoAliasHolder.payload is p else x * 3


_DYNAMO_ALIAS_MODULE = types.ModuleType("_precompile_dynamo_alias_module")
_DYNAMO_ALIAS_MODULE.payload = None


def _precompile_dynamo_module_attr_alias(x, p):
    return x * 2 if _DYNAMO_ALIAS_MODULE.payload is p else x * 3


class _PrecompileDynamoSlottedModuleBox:
    __slots__ = ("helper",)

    def __init__(self):
        self.helper = torch.nn.Linear(2, 2)


def _precompile_dynamo_slotted_box_call(x, box):
    return box.helper(x)


class _PrecompileDynamoSlottedTensorBox:
    __slots__ = ("t",)

    def __init__(self):
        self.t = torch.randn(3)


def _precompile_dynamo_slotted_tensor_default(
    x, box=_PrecompileDynamoSlottedTensorBox()
):
    return x + box.t


def _precompile_dynamo_stateful_flaky(x, mode):
    if mode == 3:
        torch._dynamo.graph_break()
    return x + mode


def _precompile_dynamo_callable_input(x, cb):
    return cb(x) + x


def _make_precompile_dynamo_closure():
    captured = torch.randn(3)

    def inner(x):
        return x + captured

    return inner


_PRECOMPILE_HELPER_DEFAULT_W = torch.randn(4)


def _precompile_dynamo_helper_with_default(x, w=_PRECOMPILE_HELPER_DEFAULT_W):
    return x * w + w


def _precompile_dynamo_calls_helper(a):
    return _precompile_dynamo_helper_with_default(a)


_PRECOMPILE_INLINED_GLOBAL_W = torch.randn(4)


def _precompile_dynamo_inlined_helper(x):
    return x * _PRECOMPILE_INLINED_GLOBAL_W


def _precompile_dynamo_calls_inlined_helper(a):
    return _precompile_dynamo_inlined_helper(a)


_PRECOMPILE_HELPER_LIST = [3, 4]


def _precompile_dynamo_helper_list_default(x, k=_PRECOMPILE_HELPER_LIST):
    return x * len(k)


def _precompile_dynamo_calls_list_helper(a, k):
    return _precompile_dynamo_helper_list_default(a) + k[0]


_PRECOMPILE_SHARED_LIST = [1, 2, 3]


def _precompile_dynamo_reads_shared_list(x, box):
    return x * len(_PRECOMPILE_SHARED_LIST) + len(box.payload)


class _PrecompileDynamoPayloadBox:
    def __init__(self, payload):
        self.payload = payload


class _PrecompileDynamoTensorPair:
    def __init__(self, a, b):
        self.a = a
        self.b = b


class _PrecompileDynamoSlottedPair:
    __slots__ = ("a", "b")

    def __init__(self, a, b):
        self.a = a
        self.b = b


class _PrecompileDynamoModuleInDict:
    # An nn.Module stored in a plain instance __dict__ (not a slot / class attr).
    def __init__(self):
        self.helper = torch.nn.Linear(2, 2)


def _precompile_dynamo_pair_sum(p):
    return p.a * 2 + p.b


def _precompile_dynamo_pair_inplace(p):
    p.a.add_(1)
    return p.a * 2 + p.b


class _PrecompileDynamoCfg:
    scale = 2
    counter = 0


def _precompile_dynamo_returns_global_class(x):
    return x + 1, _PrecompileDynamoCfg


def _precompile_dynamo_mutates_class_attr(x):
    _PrecompileDynamoCfg.counter += 1
    return x * _PrecompileDynamoCfg.scale


_PRECOMPILE_LOG: list[int] = []


def _precompile_dynamo_appends_global_list(x):
    _PRECOMPILE_LOG.append(1)
    return x + 1


_PRECOMPILE_TABLE = torch.tensor([1.0, 2.0])


class _PrecompileDynamoCfgWithMethod:
    scale = 2

    def lookup(self):
        return _PRECOMPILE_TABLE


def _precompile_dynamo_reads_class_scale(x):
    return x * _PrecompileDynamoCfgWithMethod.scale


def _precompile_dynamo_refs_sympy(x):
    return x + len(sympy.__name__)


_PRECOMPILE_HISTORY: list[int] = []


def _precompile_dynamo_reads_history(x):
    return x * (len(_PRECOMPILE_HISTORY) + 1)


def _precompile_dynamo_data_dependent(x):
    if x.sum() > 0:
        return x + 1
    return x - 1


def _precompile_dynamo_first_element(x):
    return x[0]


def _precompile_dynamo_matmul(x, w):
    return (x @ w).relu()


def _precompile_dynamo_mutating_step(xs, t):
    xs.append(1)
    return t * len(xs)


def _precompile_dynamo_helper_empty_default(x, dims=()):
    return x * (len(dims) + 1)


def _precompile_dynamo_calls_empty_default(a, dims):
    return _precompile_dynamo_helper_empty_default(a) + len(dims)


def _precompile_dynamo_helper_ellipsis_default(x, key=...):
    return x + (0 if key is ... else 1)


def _precompile_dynamo_calls_ellipsis_default(a, key):
    return _precompile_dynamo_helper_ellipsis_default(a)


def _precompile_dynamo_act(x):
    return torch.relu(x)


class _PrecompileDynamoActBox:
    def __init__(self, act):
        self.act = act


def _precompile_dynamo_calls_act(a, box):
    return _precompile_dynamo_act(a) * 2


# A custom pytree node whose context (a set) is not JSON-dumpable and which has no
# to_dumpable_context serializer, so treespec_dumps raises TypeError (distinct from the
# unregistered-namedtuple NotImplementedError path). Registered once at module load and
# used by test_unserializable_context_in_spec_still_compiles.
class _UnserializableCtxInput:
    def __init__(self, a, b):
        self.a = a
        self.b = b


_pytree.register_pytree_node(
    _UnserializableCtxInput,
    lambda n: ([n.a, n.b], {"ctx"}),
    lambda children, _ctx: _UnserializableCtxInput(children[0], children[1]),
    serialized_type_name="test_precompile._UnserializableCtxInput",
)


def _strip_artifact(cache: bytes) -> bytes:
    """Return the cache envelope with its compiled artifact removed, forcing load()
    onto the inlined (no-cache) path that JIT-compiles from python_code. Many tests
    reload the same artifact both cache-primed and stripped to check they agree."""
    blob = torch.load(io.BytesIO(cache), weights_only=True)
    blob["artifact"] = None
    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue()


def _load_dynamo_state(code: str):
    """Decode the opaque Dynamo state a tracer='dynamo' artifact embeds (the read side
    of _build_dynamo_python_source's ``_DYNAMO_STATE = ...`` line); None if absent."""
    for node in ast.parse(code).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_DYNAMO_STATE"
        ):
            return pickle.loads(base64.b64decode(ast.literal_eval(node.value)))
    return None


def _fresh_cache_env(tmp: str, **overrides: str) -> dict[str, str]:
    """Child-process env with EMPTY inductor/Triton on-disk caches, so nothing the
    parent compiled can be reused, plus TORCHDYNAMO_DISABLE stripped (a test that
    wants it passes it in ``overrides``)."""
    env = {k: v for k, v in os.environ.items() if k != "TORCHDYNAMO_DISABLE"}
    env["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(tmp, "inductor_cache")
    env["TRITON_CACHE_DIR"] = os.path.join(tmp, "triton_cache")
    env.update(overrides)
    return env


def _dynamo_serialized_guard_summary(
    code: str,
) -> list[tuple[list[str], list[str], list[str], bool]]:
    from torch._dynamo.package import load_guards_state

    state = _load_dynamo_state(code)
    summary = []
    for variant in state["variants"]:
        guards_state = load_guards_state(variant["guards_state"])
        summary.append(
            (
                [guard.create_fn_name() for guard in guards_state.output_graph.guards],
                [
                    type(guard).__name__
                    for guard in guards_state.output_graph.aotautograd_guards
                ],
                sorted(
                    source.name
                    for source in guards_state.output_graph.guard_on_key_order
                ),
                guards_state.shape_code_parts is not None,
            )
        )
    return summary


def _stateful_paths(tmp: str) -> dict[str, str]:
    """The ``artifact_path`` / ``cache_path`` kwargs a stateful capture writes under
    ``tmp`` (also splattable straight into ``stateful`` / ``load_files``)."""
    return {
        "artifact_path": os.path.join(tmp, "artifact.py"),
        "cache_path": os.path.join(tmp, "artifact.cache"),
    }


def _read_pair(paths: dict[str, str]) -> tuple[str, bytes]:
    """Read back the (python_code, cache) a stateful capture wrote to ``paths``."""
    with open(paths["artifact_path"]) as f:
        code = f.read()
    with open(paths["cache_path"], "rb") as f:
        cache = f.read()
    return code, cache


def _default_and_inlined_loaders(code: str, cache: bytes, backend: str):
    """Yield (label, loaded_fn) for the load paths a backend exposes: the default
    (cache-primed) path always, plus -- on inductor only -- the inlined path that
    strips the artifact to force JIT from python_code. The eager backend has a single
    driver, so it yields the default path alone."""
    yield "default", torch.compiler.precompile.load(code, cache)
    if backend == "inductor":
        yield "inlined", torch.compiler.precompile.load(code, _strip_artifact(cache))


# precompile drives make_fx internally, which cannot symbolically trace a
# dynamo-optimized function; the whole suite is therefore incompatible with
# PYTORCH_TEST_WITH_DYNAMO (dynamo_wrapped CI), so skip it there.
@skipIfTorchDynamo("precompile's make_fx capture is incompatible with dynamo wrapping")
@instantiate_parametrized_tests
class TestPrecompile(TestCase):
    def test_decompositions_kwarg(self):
        # The decompositions table is threaded into make_fx during capture; a
        # custom decomposition is invoked and the result still matches eager.
        called = []

        def my_relu_decomp(x):
            called.append(True)
            return (x > 0) * x

        decomps = {torch.ops.aten.relu.default: my_relu_decomp}
        m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)], decompositions=decomps
        )
        self.assertTrue(called)  # the table was used during capture

        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_constant_tensor_is_rejected(self):
        captured = torch.randn(3)
        with self.assertRaisesRegex(PrecompileError, "hard-coded"):
            torch.compiler.precompile(
                lambda x: x + captured, example_inputs=[(torch.randn(3),)]
            )

    def test_global_tensor_rejected_unlike_make_fx(self):
        # Vanilla make_fx silently bakes a referenced global tensor into the
        # GraphModule as a get_attr constant; precompile must instead error.
        from torch.fx.experimental.proxy_tensor import make_fx

        def f(x):
            return x + _GLOBAL_TENSOR

        gm = make_fx(f)(torch.randn(3))
        baked = [
            n.target
            for n in gm.graph.nodes
            if n.op == "get_attr"
            and isinstance(getattr(gm, n.target, None), torch.Tensor)
        ]
        self.assertTrue(baked, "expected vanilla make_fx to bake a tensor constant")

        with self.assertRaisesRegex(PrecompileError, "hard-coded"):
            torch.compiler.precompile(f, example_inputs=[(torch.randn(3),)])

    def test_unregistered_module_tensor_attr_is_rejected(self):
        # A plain tensor attribute (not a registered parameter/buffer) is not
        # lifted, so referencing it would bake it in -- this must error.
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 4))
                self.scale = torch.randn(4)  # plain attr, NOT a buffer/parameter

            def forward(self, x):
                return (x @ self.weight) * self.scale

        m = M().eval()
        with self.assertRaisesRegex(PrecompileError, "hard-coded"):
            torch.compiler.precompile(
                lambda model, x: model(x), example_inputs=[(m, torch.randn(2, 4))]
            )

    def test_export_and_reload_roundtrip(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)
                self.register_buffer("b2", torch.randn(3))

            def forward(self, x):
                return torch.relu(self.lin(x)) + self.b2

        m = M().eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )

        self.assertIn("Inductor output code", code)
        self.assertIn("def forward(", code)
        self.assertIn("PARAM_NAMES = ['lin.weight', 'lin.bias']", code)

        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_self_contained_exec_needs_no_cache(self):
        # python_code runs standalone with NO cache: exec it and call forward().
        # The default eager backend has no kernels; the captured graph is
        # interpreted directly from the inlined source and the cache is always
        # empty (artifact=None), so python_code is fully self-contained.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, _cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )

        ns = {"__name__": "_artifact"}
        exec(compile(code, "<artifact>", "exec"), ns)
        self.assertEqual(ns["forward"](m, x), m(x))

    @unittest.skipUnless(TEST_CUDA, "needs CUDA + Triton for the kernel cache")
    @torch._inductor.config.patch({"compile_threads": 1})
    def test_cache_reload_without_eager_static_launcher_rehydration(self):
        # A cold load should use JIT instead of eagerly rehydrating the static launcher.
        import torch._inductor.config as ind_config

        if ind_config.force_disable_caches or not ind_config.fx_graph_cache:
            self.skipTest("requires inductor FxGraphCache enabled")
        if not ind_config.use_static_cuda_launcher:
            self.skipTest("requires the static CUDA launcher")
        from torch._dynamo.utils import counters
        from torch._inductor.utils import fresh_cache

        m = (
            torch.nn.Sequential(
                torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
            )
            .eval()
            .cuda()
        )
        x = torch.randn(3, 8, device="cuda")
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        self.assertIsInstance(cache, bytes)

        with fresh_cache():
            counters.clear()
            f_c = torch.compiler.precompile.load(code, cache)
            self.assertEqual(f_c(m, x), m(x))
            self.assertEqual(
                counters["inductor"]["triton_bundler_load_static_autotuner"], 0
            )

    @unittest.skipUnless(TEST_CUDA, "needs CUDA for Triton autotuning")
    def test_cache_bundles_autotune_artifacts(self):
        from torch._inductor.utils import fresh_cache

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = torch.nn.Linear(512, 512)
                self.l2 = torch.nn.Linear(512, 512)

            def forward(self, x):
                return torch.softmax(self.l2(torch.relu(self.l1(x))), dim=-1)

        m = M().cuda().eval()
        x = torch.randn(128, 512, device="cuda")
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        with fresh_cache():
            f_c = torch.compiler.precompile.load(code, cache)
            self.assertEqual(f_c(m, x), m(x))

    def test_dtensor_subclass(self):
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("gloo not available")

        from torch.distributed.tensor import DeviceMesh, distribute_tensor, Replicate
        from torch.testing._internal.common_utils import find_free_port

        # Use a free port (a hardcoded one flakes on shared CI); patch.dict
        # restores the env so we do not leak MASTER_ADDR/MASTER_PORT to later tests.
        env = {"MASTER_ADDR": "localhost", "MASTER_PORT": str(find_free_port())}
        with mock.patch.dict(os.environ, env):
            dist.init_process_group("gloo", rank=0, world_size=1)
            try:
                mesh = DeviceMesh("cpu", list(range(1)))
                m = torch.nn.Linear(4, 3).eval()
                for name, p in list(m.named_parameters()):
                    setattr(
                        m,
                        name,
                        torch.nn.Parameter(
                            distribute_tensor(p.detach(), mesh, [Replicate()])
                        ),
                    )
                x = distribute_tensor(torch.randn(5, 4), mesh, [Replicate()])
                ref = m(x)

                code, cache = torch.compiler.precompile(
                    lambda model, x: model(x), example_inputs=[(m, x)]
                )
                # Subclass handling is via our own protocol-based driver, not embedded
                # AOTAutograd wrapper source.
                self.assertIn("__tensor_unflatten__", code)
                self.assertNotIn("subclass_wrapper", code)

                # load() takes the bundled-artifact path (real AOTAutograd runtime).
                f_c = torch.compiler.precompile.load(code, cache)
                self.assertEqual(f_c(m, x).to_local(), ref.to_local())

                # Also exercise the standalone driver (the generated python, no cache):
                # subclass inputs/outputs handled by the inlined recipes via
                # __tensor_flatten__/__tensor_unflatten__.
                ns = {"__name__": "_dt"}
                exec(compile(code, "<dt>", "exec"), ns)
                self.assertEqual(ns["forward"](m, x).to_local(), ref.to_local())
            finally:
                dist.destroy_process_group()

    def test_cache_holds_only_artifact(self):
        # The cache is purely an acceleration: the only COMPILED blob it carries is the
        # ``artifact`` (no weights, no calling-convention metadata -- that lives in
        # python_code, the single source of truth, and load() parses it back from
        # there). The envelope additionally carries a lightweight format/version/backend
        # integrity tag (plain str/int), which load() verifies.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )

        from torch._precompile import _CACHE_FORMAT, _CACHE_VERSION

        blob = torch.load(io.BytesIO(cache), weights_only=False)
        # The artifact is the only compiled blob; the rest is the integrity tag (the
        # format/version/backend tag plus a code_hash binding the cache to its python_code).
        self.assertEqual(
            set(blob), {"artifact", "format", "version", "backend", "code_hash"}
        )
        self.assertEqual(blob["format"], _CACHE_FORMAT)
        self.assertEqual(blob["version"], _CACHE_VERSION)
        self.assertEqual(blob["backend"], "inductor")
        self.assertIsInstance(blob["artifact"], bytes)
        # The calling convention is recoverable from python_code alone.
        from torch._precompile import _parse_artifact_metadata

        meta = _parse_artifact_metadata(code)
        self.assertEqual(meta["BACKEND"], "inductor")
        self.assertEqual(meta["MODULE_POSITIONS"], [0])

        # load() works using metadata from python_code + artifact from the cache.
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_inlined_fallback_when_artifact_absent(self):
        # When the cache holds no serialized artifact, load() falls back to
        # executing the inlined python (recompiling kernels). Force that branch by
        # stripping the artifact and check it still matches eager; this also
        # exercises the self-contained inlined path (JIT from inlined source).
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )

        blob = torch.load(io.BytesIO(cache), weights_only=False)
        self.assertIsNotNone(blob["artifact"])

        f_c = torch.compiler.precompile.load(code, _strip_artifact(cache))
        self.assertEqual(f_c(m, x), m(x))

    def test_cache_envelope_is_weights_only_safe(self):
        # The cache is a plain {"artifact": bytes, "format"/"version"/"backend": ...}
        # envelope of only str/int/bytes: it loads with the safe unpickler
        # (weights_only=True). The executable part is the inner artifact bytes, fed to
        # load_cache_artifacts inside load() to prime the inductor cache -- that (plus the
        # subsequent exec of python_code) is the code-execution step, not this outer load.
        # The integrity tag is present and correct (and itself weights_only-safe).
        from torch._precompile import _CACHE_FORMAT, _CACHE_VERSION

        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        _code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        blob = torch.load(io.BytesIO(cache), weights_only=True)  # must not raise
        self.assertEqual(
            set(blob), {"artifact", "format", "version", "backend", "code_hash"}
        )
        self.assertEqual(blob["format"], _CACHE_FORMAT)
        self.assertEqual(blob["version"], _CACHE_VERSION)
        self.assertEqual(blob["backend"], "inductor")
        # code_hash is a plain str (sha256 hexdigest), so the envelope stays
        # weights_only-safe even with this added key.
        self.assertIsInstance(blob["code_hash"], str)

    def test_wrong_param_count_model_rejected(self):
        # Invariant 2: a runtime model whose param/buffer count differs from the
        # traced model is rejected with a clear error rather than an opaque inner
        # failure. This exercises the default eager load path, which execs
        # python_code (the eager cache carries no artifact).
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)

        bigger = torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.Linear(4, 3)
        ).eval()
        with self.assertRaisesRegex(PrecompileError, "structurally identical"):
            f_c(bigger, x)

    def test_wrong_param_count_rejected_inlined(self):
        # The same guard fires on the inlined (no-cache) path with the same exception
        # type as the cached path (PrecompileError): strip the artifact so load()
        # execs python_code, then call with a structurally different model.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, _strip_artifact(cache))

        bigger = torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.Linear(4, 3)
        ).eval()
        with self.assertRaisesRegex(PrecompileError, "structurally identical"):
            f_c(bigger, x)

    def test_runtime_input_structure_mismatch_rejected(self):
        # Invariant 3: a runtime input whose pytree structure differs from the traced
        # example (here a list where a bare tensor was traced) is rejected via the
        # IN_SPEC check, rather than silently flattening to the wrong leaves.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "different structure"):
            f_c(m, [x, x])

    def test_unserializable_in_spec_still_compiles(self):
        # A runtime input whose pytree TreeSpec is not JSON-serializable (an unregistered
        # collections.namedtuple) must still compile/run on the default eager backend:
        # IN_SPEC degrades to None and the structure check is skipped rather than
        # hard-failing.
        import collections

        P = collections.namedtuple("P", ["x", "y"])
        m = torch.nn.Linear(4, 3).eval()
        inp = P(torch.randn(5, 4), torch.randn(5, 4))
        code, cache = torch.compiler.precompile(
            lambda model, p: model(p.x + p.y), example_inputs=[(m, inp)]
        )
        self.assertIn("IN_SPEC = None", code)
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, inp), m(inp.x + inp.y))

    def test_unserializable_context_in_spec_still_compiles(self):
        # A registered pytree node whose context is not JSON-dumpable makes
        # treespec_dumps raise TypeError (not NotImplementedError); IN_SPEC must still
        # degrade to None rather than crashing precompile.
        m = torch.nn.Linear(4, 3).eval()
        inp = _UnserializableCtxInput(torch.randn(5, 4), torch.randn(5, 4))
        code, cache = torch.compiler.precompile(
            lambda model, h: model(h.a + h.b), example_inputs=[(m, inp)]
        )
        self.assertIn("IN_SPEC = None", code)
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, inp), m(inp.a + inp.b))

    def test_unserializable_out_spec_hard_fails(self):
        # OUT_SPEC is load-bearing (the driver rebuilds fn's output via tree_unflatten),
        # so unlike IN_SPEC it CANNOT degrade to None. An fn that RETURNS an unregistered
        # collections.namedtuple has a non-JSON-serializable output TreeSpec and must
        # raise a clear PrecompileError rather than leaking a raw pytree error.
        import collections

        Out = collections.namedtuple("Out", ["a", "b"])
        with self.assertRaisesRegex(
            PrecompileError, "cannot serialize the output structure"
        ):
            torch.compiler.precompile(
                lambda x: Out(x + 1, x + 2), example_inputs=[(torch.randn(4),)]
            )

    @parametrize("backend", ("inductor", "eager"))
    def test_input_leaf_count_mismatch_rejected_when_spec_unserializable(self, backend):
        # When IN_SPEC degrades to None the structural in_spec check is skipped; a runtime
        # input flattening to a DIFFERENT leaf count must still raise a clean
        # PrecompileError (not a raw zip/unpack error) on the live and eager-inlined paths.
        m = torch.nn.Linear(4, 3).eval()
        inp = _UnserializableCtxInput(torch.randn(5, 4), torch.randn(5, 4))
        code, cache = torch.compiler.precompile(
            lambda model, h: model(h.a + h.b),
            example_inputs=[(m, inp)],
            backend=backend,
        )
        self.assertIn("IN_SPEC = None", code)
        f = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "flattened to"):
            f(m, torch.randn(5, 4))  # one leaf vs the traced two

    def test_user_input_error_precedes_structural_error(self):
        # All three load paths run the user-input checks BEFORE the structural model-name
        # check, so a call violating BOTH (wrong dtype and a different model) reports the
        # user-input (dtype) error, keeping the first-reported error consistent.
        m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)

        class B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l0 = torch.nn.Linear(4, 4)
                self.l1 = torch.nn.Linear(4, 3)

            def forward(self, t):
                return self.l1(self.l0(t))

        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        f_i = torch.compiler.precompile.load(code, _strip_artifact(cache))
        code_e, cache_e = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)], backend="eager"
        )
        f_e = torch.compiler.precompile.load(code_e, cache_e)
        for f in (f_c, f_i, f_e):
            with self.assertRaisesRegex(PrecompileError, "dtype"):
                f(
                    B(), x.double()
                )  # wrong model AND wrong dtype -> dtype reported first

    @parametrize("backend", ("inductor", "eager"))
    def test_unserializable_out_spec_rejected(self, backend):
        # OUT_SPEC is load-bearing (the driver rebuilds fn's output via tree_unflatten),
        # so unlike IN_SPEC it cannot degrade to None: a fn returning an unregistered
        # namedtuple must fail with a clear PrecompileError, not a raw pytree error, on
        # both backends. A registered namedtuple output round-trips fine.
        import collections

        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        NT = collections.namedtuple("NT", ["p", "q"])
        with self.assertRaisesRegex(PrecompileError, "output structure"):
            torch.compiler.precompile(
                lambda model, xx: NT(model(xx), model(xx) + 1),
                example_inputs=[(m, x)],
                backend=backend,
            )
        # A registered namedtuple output serializes and round-trips on both backends.
        # Registration mutates the process-global pytree registry, so deregister it on
        # cleanup rather than leaking the node into later tests.
        RNT = collections.namedtuple("RNT", ["p", "q"])
        _pytree._register_namedtuple(RNT, serialized_type_name="test_precompile.RNT")
        self.addCleanup(_pytree._deregister_pytree_node, RNT)
        ref = (m(x), m(x) + 1)
        code, cache = torch.compiler.precompile(
            lambda model, xx: RNT(model(xx), model(xx) + 1),
            example_inputs=[(m, x)],
            backend=backend,
        )
        out = torch.compiler.precompile.load(code, cache)(m, x)
        self.assertEqual((out.p, out.q), ref)

    def test_cached_and_inlined_paths_agree(self):
        # Both load paths exec the SAME inlined driver in python_code; the only difference
        # is whether the cache primed the kernels first (warm) or not (cold JIT). They must
        # produce identical results -- cross-check via identical scattered grads from a
        # cache-primed load and a cache-stripped (artifact=None) load of the SAME artifact,
        # with multiple modules AND a tied weight across two of them (the case where an
        # ordering divergence in the embedded _extract_param_buffers would show).
        torch.manual_seed(0)
        a = torch.nn.Linear(4, 4, bias=False)
        b = torch.nn.Linear(4, 4, bias=False)
        b.weight = a.weight  # tie across two distinct module args
        c = torch.nn.Linear(4, 3)
        loss_fn = torch.nn.MSELoss()
        x = torch.randn(2, 4)
        target = torch.randn(2, 3)

        def step(ma, mb, mc, x, target):
            loss_fn(mc(mb(torch.relu(ma(x)))), target).backward()

        code, cache = torch.compiler.precompile(
            step, example_inputs=[(a, b, c, x, target)]
        )

        def grads(ms):
            return [p.grad for m in ms for p in m.parameters()]

        # deepcopy the three together so the a/b weight tie is preserved.
        ca, cb, cc = copy.deepcopy((a, b, c))
        torch.compiler.precompile.load(code, cache)(
            ca, cb, cc, x, target
        )  # cached path

        ia, ib, ic = copy.deepcopy((a, b, c))
        torch.compiler.precompile.load(code, _strip_artifact(cache))(
            ia, ib, ic, x, target
        )  # inlined

        for cg, ig in zip(grads((ca, cb, cc)), grads((ia, ib, ic))):
            self.assertEqual(cg, ig)

    def test_eager_param_ordering_agrees_with_inductor(self):
        # Both backends now emit the same _extract_param_buffers (from
        # torch._precompile_driver), which must stay in sync with
        # torch._precompile._intern_param_buffers. The test above cross-checks only the
        # cached vs inductor-inlined paths; cross-check the EAGER backend too, on the same
        # multi-module + tied-weight + backward step, so an ordering divergence in the
        # shared driver shows as a scattered-grad mismatch against the inductor cached path.
        torch.manual_seed(0)
        a = torch.nn.Linear(4, 4, bias=False)
        b = torch.nn.Linear(4, 4, bias=False)
        b.weight = a.weight  # tie across two distinct module args
        c = torch.nn.Linear(4, 3)
        loss_fn = torch.nn.MSELoss()
        x = torch.randn(2, 4)
        target = torch.randn(2, 3)

        def step(ma, mb, mc, x, target):
            loss_fn(mc(mb(torch.relu(ma(x)))), target).backward()

        def grads(ms):
            return [p.grad for m in ms for p in m.parameters()]

        # deepcopy the three together so the a/b weight tie is preserved.
        icode, icache = torch.compiler.precompile(
            step, example_inputs=[(a, b, c, x, target)]
        )
        ia, ib, ic = copy.deepcopy((a, b, c))
        torch.compiler.precompile.load(icode, icache)(
            ia, ib, ic, x, target
        )  # inductor cached path

        ecode, ecache = torch.compiler.precompile(
            step, example_inputs=[(a, b, c, x, target)], backend="eager"
        )
        ea, eb, ec = copy.deepcopy((a, b, c))
        torch.compiler.precompile.load(ecode, ecache)(
            ea, eb, ec, x, target
        )  # eager path

        ind_grads = grads((ia, ib, ic))
        eager_grads = grads((ea, eb, ec))
        self.assertEqual(len(ind_grads), len(eager_grads))
        for ig, eg in zip(ind_grads, eager_grads):
            self.assertEqual(ig, eg)

    def test_non_module_at_module_position_rejected(self):
        # Passing a non-nn.Module where the traced fn took a module yields a clear
        # PrecompileError citing invariant 2, not a bare AttributeError.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "must be the nn.Module"):
            f_c(x, x)  # tensor at the module slot

    def test_wrong_arg_count_rejected(self):
        # A runtime call with the wrong number of positional args raises a clear
        # PrecompileError (invariant 2) -- not a raw IndexError -- on all three load
        # paths, including when a module is at a non-zero position (where args[i] would
        # otherwise index past the short args tuple).
        m = torch.nn.Linear(4, 3)
        x = torch.randn(2, 4)
        # Module at position 1 (so a missing trailing arg would index past args).
        code, cache = torch.compiler.precompile(
            lambda xx, model: model(xx), example_inputs=[(x, m)]
        )
        inlined_cache = _strip_artifact(cache)  # force the inlined path
        ecode, ecache = torch.compiler.precompile(
            lambda xx, model: model(xx), example_inputs=[(x, m)], backend="eager"
        )
        loaders = {
            "cached": torch.compiler.precompile.load(code, cache),
            "inlined": torch.compiler.precompile.load(code, inlined_cache),
            "eager": torch.compiler.precompile.load(ecode, ecache),
        }
        for label, f_c in loaders.items():
            with self.subTest(path=label):
                with self.assertRaisesRegex(PrecompileError, "expected 2 positional"):
                    f_c(x)  # too few (omits the module arg)
                with self.assertRaisesRegex(PrecompileError, "expected 2 positional"):
                    f_c(x, m, x)  # too many
                self.assertEqual(f_c(x, m), m(x))  # correct arity still works

    def test_buffer_requiring_grad_rejected(self):
        # A registered buffer with requires_grad=True that receives a gradient is not
        # harvested (only params are), so precompile rejects it rather than silently
        # dropping the grad.
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("b", torch.randn(4, requires_grad=True))

            def forward(self, x):
                return (x * self.b).sum()

        m = M()
        x = torch.randn(4)
        with self.assertRaisesRegex(PrecompileError, "buffer received a gradient"):
            torch.compiler.precompile(
                lambda model, x: model(x).backward(), example_inputs=[(m, x)]
            )

    def test_user_input_requiring_grad_rejected(self):
        # Sibling of the buffer guard: a requires_grad USER INPUT (not a param) that
        # receives a gradient during the traced backward is not harvested (only params
        # are), so precompile rejects it rather than silently dropping the grad.
        x = torch.randn(4, requires_grad=True)
        with self.assertRaisesRegex(PrecompileError, "user input received a gradient"):
            torch.compiler.precompile(
                lambda t: (t * t).sum().backward(), example_inputs=[(x,)]
            )

    def test_control_flow_subgraph_rejected(self):
        # torch.cond captures as a HOP with get_attr subgraph submodules, which the
        # standalone artifact cannot inline; reject it at capture with a clear message.
        def f(x):
            return torch.cond(x.sum() > 0, lambda t: t + 1, lambda t: t - 1, (x,))

        with self.assertRaisesRegex(PrecompileError, "control-flow subgraph"):
            torch.compiler.precompile(f, example_inputs=[(torch.randn(4),)])

    def test_load_falls_back_when_cache_unreconstructable(self):
        # The cache is only an acceleration; python_code always runs standalone. A
        # corrupt / stale cache must degrade to the inlined JIT path, not crash.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        self.assertIsNotNone(blob["artifact"])
        blob["artifact"] = b"corrupt-not-a-real-artifact"
        buf = io.BytesIO()
        torch.save(blob, buf)

        f_c = torch.compiler.precompile.load(code, buf.getvalue())  # must not raise
        self.assertEqual(f_c(m, x), m(x))

    def test_load_falls_back_on_corrupt_cache_envelope(self):
        # Not just a bad inner artifact -- a corrupt/truncated cache ENVELOPE (not even
        # a valid torch.save blob) must also degrade to the inlined python_code path,
        # since the cache is purely an acceleration.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, _cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(
            code, b"not-a-torch-save-blob"
        )  # must not raise
        self.assertEqual(f_c(m, x), m(x))

    def test_load_invalid_python_code_rejected(self):
        # load() surfaces a clear PrecompileError (not a raw SyntaxError) when
        # python_code is not valid Python.
        buf = io.BytesIO()
        torch.save({"artifact": None}, buf)
        with self.assertRaisesRegex(PrecompileError, "not valid Python"):
            torch.compiler.precompile.load("def (:::", buf.getvalue())

    def test_untrusted_input_warning_fires_per_load(self):
        # The trust warning is emitted PER load (not warning_once) via log.warning on the
        # torch._precompile logger: load() warns before any cache processing and then
        # always execs python_code, whether or not the cache primed the kernels first.
        # Calling load() TWICE must fire the untrusted-input warning on BOTH calls,
        # locking in per-load behavior rather than once-per-process.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        # Cached path (inductor): the exec of python_code warns about untrusted input.
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)]
        )
        for _ in range(2):
            with self.assertLogs("torch._precompile", level="WARNING") as cm:
                torch.compiler.precompile.load(code, cache)
            self.assertTrue(
                any("untrusted" in line.lower() for line in cm.output),
                f"cached load did not warn about untrusted input: {cm.output}",
            )
        # Eager backend (empty cache, nothing to prime): load() still warns about the
        # untrusted input up front and EXECs python_code, every load.
        ecode, ecache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)], backend="eager"
        )
        for _ in range(2):
            with self.assertLogs("torch._precompile", level="WARNING") as cm:
                torch.compiler.precompile.load(ecode, ecache)
            self.assertTrue(
                any("untrusted" in line.lower() for line in cm.output),
                f"inlined load did not warn about untrusted input: {cm.output}",
            )
            self.assertTrue(
                any("EXEC" in line for line in cm.output),
                f"inlined load did not warn about exec'ing python_code: {cm.output}",
            )

    def test_no_compute_graph_rejected_inductor(self):
        # The inductor backend produces no runnable module for a graph with no compute
        # to lower -- one that returns inputs or Python constants unchanged (a constant,
        # a bare passthrough, or an alias like .detach()). Reject with a clear
        # PrecompileError rather than a raw "found 0 runnable modules" RuntimeError. The
        # eager backend handles these (the contract is otherwise identical).
        x = torch.randn(4)
        cases = {
            "constant": lambda xx: 7,
            "passthrough": lambda xx: xx,
            "detach_alias": lambda xx: xx.detach(),
        }
        for label, fn in cases.items():
            with self.subTest(case=label):
                with self.assertRaisesRegex(PrecompileError, "no compute"):
                    torch.compiler.precompile(fn, example_inputs=[(x,)])
        # The eager backend handles a passthrough and a constant fn.
        code, cache = torch.compiler.precompile(
            lambda xx: xx, example_inputs=[(x,)], backend="eager"
        )
        self.assertEqual(torch.compiler.precompile.load(code, cache)(x), x)
        code, cache = torch.compiler.precompile(
            lambda xx: 7, example_inputs=[(x,)], backend="eager"
        )
        self.assertEqual(torch.compiler.precompile.load(code, cache)(x), 7)

    def test_same_count_different_structure_rejected(self):
        # Invariant 2: the structural check now compares the baked PARAM_NAMES /
        # BUFFER_NAMES against the runtime model's extracted param/buffer names, so a
        # same-count-but-different-structure (here, differently-NAMED submodules) model
        # is REJECTED rather than silently running the traced graph with the wrong
        # weights. Both the cached and the inlined (artifact-stripped) load paths fire.
        a = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).eval()
        x = torch.randn(2, 4)
        code, cache = torch.compiler.precompile(
            lambda m, x: m(x), example_inputs=[(a, x)]
        )
        # The traced names come from the Sequential (``0.weight``, ``1.weight`` ...).
        self.assertIn(
            "PARAM_NAMES = ['0.weight', '0.bias', '1.weight', '1.bias']", code
        )

        class B(torch.nn.Module):  # same 4 params (same count/shapes), different names
            def __init__(self):
                super().__init__()
                self.l0 = torch.nn.Linear(4, 4)
                self.l1 = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.l0(x) + self.l1(x)

        b = B().eval()
        loaders = {
            "cached": torch.compiler.precompile.load(code, cache),
            "inlined": torch.compiler.precompile.load(code, _strip_artifact(cache)),
        }
        for label, f_c in loaders.items():
            with self.subTest(path=label):
                with self.assertRaisesRegex(
                    PrecompileError, "do not match the traced model"
                ):
                    f_c(b, x)

    def test_same_count_different_structure_rejected_eager(self):
        # The eager driver's _check_structure rejects a same-param-COUNT but
        # different-NAME model (here differently-named submodules) rather than
        # silently running the traced graph with the wrong weights (invariant 2).
        # What's distinct from test_wrong_param_count_model_rejected above is the
        # INPUT -- same count / different name, not a count mismatch.
        a = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).eval()
        x = torch.randn(2, 4)
        code, cache = torch.compiler.precompile(
            lambda m, x: m(x), example_inputs=[(a, x)], backend="eager"
        )
        self.assertIn(
            "PARAM_NAMES = ['0.weight', '0.bias', '1.weight', '1.bias']", code
        )

        class B(torch.nn.Module):  # same 4 params (same count/shapes), different names
            def __init__(self):
                super().__init__()
                self.l0 = torch.nn.Linear(4, 4)
                self.l1 = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.l0(x) + self.l1(x)

        b = B().eval()
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "do not match the traced model"):
            f_c(b, x)

    # Input mutation, output aliasing, tensor subclasses, and functionalized RNG are
    # SUPPORTED: the inductor backend lowers through aot_autograd.compile_to_python,
    # which composes AOTAutograd's own codegen'd prelude/epilogue into the artifact.
    # Only effectful ops are rejected up front (see test_effectful_op_unsupported).

    def test_effectful_op_unsupported(self):
        # Effectful custom ops are rejected up front by _assert_supported, which
        # detects the with_effects HOP in the captured graph -- the effect cannot
        # be lowered to standalone source, so capture fails cleanly.
        from torch._higher_order_ops.effects import _EffectType, _register_effectful_op
        from torch.library import _scoped_library

        with _scoped_library("mlprecompile", "FRAGMENT") as lib:
            lib.define("eff(Tensor x) -> Tensor")
            lib.impl("eff", lambda x: x + 1.0, "CompositeExplicitAutograd")
            lib.impl("eff", lambda x: torch.empty_like(x), "Meta")
            op = torch.ops.mlprecompile.eff.default
            _register_effectful_op(op, _EffectType.ORDERED)
            try:
                with self.assertRaisesRegex(
                    PrecompileError, "effectful op.*not supported yet"
                ):
                    torch.compiler.precompile(
                        lambda a: torch.ops.mlprecompile.eff(a),
                        example_inputs=[(torch.randn(4),)],
                    )
            finally:
                _register_effectful_op(op, None)

    def test_public_api_surface(self):
        # precompile is a public API under the compiler namespace
        # (torch.compiler.precompile), with a load method and a public error type;
        # it is deliberately NOT a top-level torch.* verb.
        self.assertIn("precompile", torch.compiler.__all__)
        self.assertNotIn("precompile", torch.__all__)
        # __all__ membership and the attribute itself are independent, so lock in
        # removal of the top-level entry point too (re-adding the re-export without
        # touching __all__ would silently resurrect torch.precompile).
        self.assertFalse(hasattr(torch, "precompile"))
        self.assertTrue(callable(torch.compiler.precompile))
        self.assertTrue(callable(torch.compiler.precompile.load))
        self.assertTrue(callable(torch.compiler.precompile.load_files))
        self.assertTrue(callable(torch.compiler.precompile.stateful))
        self.assertIs(torch.compiler.precompile.PrecompileError, PrecompileError)
        self.assertIsInstance(torch.compiler.precompile.PrecompileState, type)
        self.assertIsInstance(torch.compiler.precompile.PrecompileStateSummary, type)
        # The public location: test_public_bindings.test_correct_module_names also
        # enforces this for every torch.compiler.__all__ member.
        self.assertEqual(torch.compiler.precompile.__module__, "torch.compiler")

    def test_backend_invalid_raises(self):
        a, b = torch.randn(4, 4), torch.randn(4, 4)
        with self.assertRaisesRegex(
            ValueError, "backend must be 'inductor' or 'eager'"
        ):
            torch.compiler.precompile(
                lambda x, y: x + y, example_inputs=[(a, b)], backend="nope"
            )

    def test_tracer_default_and_explicit_make_fx(self):
        # tracer defaults to "make_fx"; passing it explicitly is equivalent and works.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        for kwargs in ({}, {"tracer": "make_fx"}):
            code, cache = torch.compiler.precompile(
                lambda model, xx: model(xx), example_inputs=[(m, x)], **kwargs
            )
            self.assertEqual(torch.compiler.precompile.load(code, cache)(m, x), m(x))

    @parametrize("num_examples", [0, 2])
    def test_make_fx_requires_one_example_input(self, num_examples):
        x = torch.randn(4)
        message = "at least one" if num_examples == 0 else "exactly one"
        with self.assertRaisesRegex(ValueError, message):
            torch.compiler.precompile(
                lambda t: t + 1,
                example_inputs=[(x,)] * num_examples,
                backend="eager",
            )

    def test_positional_example_inputs_remain_supported(self):
        x = torch.randn(4)
        y = torch.randn(4)
        code, cache = torch.compiler.precompile(
            lambda left, right: left + right, x, y, backend="eager"
        )
        self.assertEqual(torch.compiler.precompile.load(code, cache)(x, y), x + y)

        with self.assertRaisesRegex(TypeError, "both positional examples and"):
            torch.compiler.precompile(
                lambda t: t + 1,
                x,
                example_inputs=[(x,)],
                backend="eager",
            )

    def test_positional_example_inputs_with_tracer_dynamo(self):
        # The BC positional-example form composes with tracer='dynamo' (the
        # positional examples become the single example tuple).
        x = torch.randn(3)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_torch_sin, x, tracer="dynamo", backend="eager"
        )
        self.assertIn('TRACER = "dynamo"', code)
        self.assertEqual(torch.compiler.precompile.load(code, cache)(x), torch.sin(x))

    def test_zero_argument_call_remains_supported(self):
        code, cache = torch.compiler.precompile(lambda: 3, backend="eager")
        self.assertEqual(torch.compiler.precompile.load(code, cache)(), 3)

    @parametrize("tracer", ("make_fx", "dynamo"))
    def test_example_inputs_require_tuples(self, tracer):
        with self.assertRaisesRegex(TypeError, "positional-argument tuples"):
            torch.compiler.precompile(
                lambda t: t + 1,
                example_inputs=[[torch.randn(4)]],
                tracer=tracer,
                backend="eager",
            )

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_recompiles_to_dynamic_graph(self):
        examples = [(torch.randn(size, 4),) for size in (2, 3, 5)]
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=examples,
            tracer="dynamo",
        )

        self.assertIn('TRACER = "dynamo"', code)
        self.assertIn("VARIANT_COUNT = 2", code)
        self.assertIn("GRAPH_COUNT = 2", code)
        self.assertIn("DYNAMIC_GRAPH_COUNT = 1", code)
        self.assertIn("Inductor output code", code)
        self.assertIn("Guard trees and transformed Dynamo bytecode", code)
        self.assertIn("_DYNAMO_BACKEND_SOURCES = (", code)
        self.assertIn(
            "# Generated by torch._functorch.aot_autograd.compile_to_python", code
        )
        # Serving numerics for the dynamic variant live in the device-generic
        # TestPrecompileNumerics.test_tracer_dynamo_dynamic_numerics.
        loaded = torch.compiler.precompile.load(code, cache)
        x = torch.randn(2, 4)
        self.assertEqual(loaded(x), _precompile_dynamo_dynamic(x))

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_keeps_invariant_input_guards(self):
        # The dynamic variant still carries the input-derived guards its capture
        # implied: the else branch (shape[0] != 1) taken by every example, plus
        # dynamic shapes' 0/1 specialization. Pin that a DYNAMIC variant exists,
        # then show a size it accepts serves while size 1 -- which the retained
        # guard rejects -- misses. With the guard dropped, size 1 would wrongly
        # be served x+1 by the dynamic graph instead of raising; with dynamic=False
        # there would be no dynamic variant at all and the first assert would fail.
        examples = [(torch.randn(size, 4),) for size in (2, 3, 5)]
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic_branch,
            example_inputs=examples,
            tracer="dynamo",
            backend="eager",
        )
        self.assertIn("DYNAMIC_GRAPH_COUNT = 1", code)
        loaded = torch.compiler.precompile.load(code, cache)
        big = torch.randn(9, 4)  # a size the dynamic variant accepts
        self.assertEqual(loaded(big), _precompile_dynamo_dynamic_branch(big))
        with self.assertRaisesRegex(PrecompileError, "no captured Dynamo variant"):
            loaded(torch.randn(1, 4))

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_keeps_tensor_metadata_guards(self):
        # The dynamic variant retains its per-input TENSOR_MATCH metadata guard
        # (dtype), not merely a size guard. Pin that a DYNAMIC variant exists,
        # then show a size it accepts serves at the captured dtype but MISSES at
        # a different dtype of the SAME accepted size -- so a plain size-guard
        # miss cannot explain the raise, and dropping the dtype guard would serve
        # the float64 input instead of raising.
        examples = [(torch.randn(size, 4),) for size in (2, 3)]
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=examples,
            tracer="dynamo",
        )
        self.assertIn("DYNAMIC_GRAPH_COUNT = 1", code)
        loaded = torch.compiler.precompile.load(code, cache)
        x = torch.randn(7, 4)  # a size the dynamic variant accepts
        self.assertEqual(loaded(x), _precompile_dynamo_dynamic(x))
        with self.assertRaisesRegex(PrecompileError, "no captured Dynamo variant"):
            loaded(
                x.double()
            )  # same accepted size, dtype mismatch -> TENSOR_MATCH miss

    def test_tracer_dynamo_capture_preserves_existing_compile_entries(self):
        from torch._dynamo.testing import CompileCounter

        torch._dynamo.reset()
        counter = CompileCounter()
        compiled = torch.compile(
            _precompile_dynamo_dynamic, backend=counter, dynamic=False
        )
        x = torch.randn(4)
        self.assertEqual(compiled(x), _precompile_dynamo_dynamic(x))
        self.assertEqual(counter.frame_count, 1)

        torch.compiler.precompile(
            _precompile_dynamo_scalar,
            example_inputs=[(x, 2)],
            tracer="dynamo",
            backend="eager",
        )
        self.assertEqual(compiled(x), _precompile_dynamo_dynamic(x))
        self.assertEqual(counter.frame_count, 1)

    def test_tracer_dynamo_capture_isolated_from_same_function_cache(self):
        from torch._dynamo.testing import CompileCounter

        torch._dynamo.reset()
        counter = CompileCounter()
        compiled = torch.compile(
            _precompile_dynamo_dynamic, backend=counter, dynamic=False
        )
        x = torch.randn(4)
        self.assertEqual(compiled(x), _precompile_dynamo_dynamic(x))

        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        self.assertIn("GRAPH_COUNT = 1", code)
        self.assertEqual(
            torch.compiler.precompile.load(code, cache)(x),
            _precompile_dynamo_dynamic(x),
        )
        # Re-running the earlier torch.compile'd fn must hit its existing cache
        # entry (frame_count stays 1): the capture ran isolated from it.
        self.assertEqual(compiled(x), _precompile_dynamo_dynamic(x))
        self.assertEqual(counter.frame_count, 1)

    def test_tracer_dynamo_filters_torch_global_guards_before_serializing(self):
        from torch._dynamo.package import load_guards_state

        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_torch_sin,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        # The serialized variants must carry no global-rooted guard: the torch
        # module guard this fn produces is environment-only and minimized away.
        for variant in _load_dynamo_state(code)["variants"]:
            guards_state = load_guards_state(variant["guards_state"])
            global_guards = [
                guard.name
                for guard in guards_state.output_graph.guards
                if (guard.name or "").startswith("G[")
            ]
            self.assertEqual(global_guards, [])
        self.assertEqual(torch.compiler.precompile.load(code, cache)(x), torch.sin(x))

    def test_tracer_dynamo_eager_backward_is_live_autograd(self):
        # Both backends capture grad-enabled and infer differentiability from
        # requires_grad inputs. The eager backend's backward is LIVE autograd
        # through the emitted ops (not captured), like
        # torch.compile(backend="eager") -- so it matches eager gradients with
        # no tangent-pattern specialization.
        x = torch.randn(4, requires_grad=True)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_torch_sin,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        out = torch.compiler.precompile.load(code, cache)(x)
        self.assertTrue(out.requires_grad)
        out.sum().backward()
        x_ref = x.detach().clone().requires_grad_()
        _precompile_dynamo_torch_sin(x_ref).sum().backward()
        self.assertEqual(x.grad, x_ref.grad)

    def test_tracer_dynamo_eager_backward_supports_higher_order(self):
        # The eager backend's backward is live autograd, so grad-of-grad works
        # exactly as in eager: pin the documented "any tangent pattern and
        # higher-order grad work" half of the backward-order contract.
        x = torch.randn(4, requires_grad=True)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_torch_sin,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        out = torch.compiler.precompile.load(code, cache)(x)
        (grad,) = torch.autograd.grad(out.sum(), x, create_graph=True)
        (grad2,) = torch.autograd.grad(grad.sum(), x)
        x_ref = x.detach().clone().requires_grad_()
        ref = _precompile_dynamo_torch_sin(x_ref)
        (ref_grad,) = torch.autograd.grad(ref.sum(), x_ref, create_graph=True)
        (ref_grad2,) = torch.autograd.grad(ref_grad.sum(), x_ref)
        self.assertEqual(grad, ref_grad)
        self.assertEqual(grad2, ref_grad2)

    @parametrize("backend", ["eager", "inductor"])
    def test_tracer_dynamo_serves_eager_semantics_under_no_grad(self, backend):
        # The driver dispatches under the pinned capture-time grad mode (the
        # serialized guards require it), but an ambient no_grad at call time
        # gets eager semantics: freshly created outputs are detached, while
        # inputs passed through are returned untouched -- exactly what eager
        # and torch.compile produce under no_grad.
        x = torch.randn(4, requires_grad=True)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_sin_and_passthrough,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend=backend,
        )
        loaded = torch.compiler.precompile.load(code, cache)
        with torch.no_grad():
            out, passthrough = loaded(x)
            ref, _ = _precompile_dynamo_sin_and_passthrough(x)
        self.assertFalse(out.requires_grad)
        self.assertIsNone(out.grad_fn)
        self.assertEqual(out, ref)
        self.assertIs(passthrough, x)
        self.assertTrue(passthrough.requires_grad)
        # A grad-mode call of the same loaded artifact stays differentiable.
        out, _ = loaded(x)
        self.assertTrue(out.requires_grad)

    def test_tracer_dynamo_rejects_inference_mode_serving(self):
        x = torch.randn(4, requires_grad=True)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_torch_sin,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "inference_mode"):
            with torch.inference_mode():
                loaded(x)

    def test_tracer_dynamo_rejects_identity_check_against_global_tensor(self):
        # The global tensor is never a graph input (only its identity is
        # compared), so the placeholder check cannot see it; Dynamo still
        # guards on it (TENSOR_MATCH), and dropping that environment guard is
        # rejected because the artifact could not reproduce the tensor.
        with self.assertRaisesRegex(
            PrecompileError, r"uses the tensor\(s\) \[\"G\['_DYNAMO_INPUT_GLOBAL'\]\"\]"
        ):
            torch.compiler.precompile(
                _precompile_dynamo_input_global_identity,
                example_inputs=[(torch.randn(3),)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_tensor_global(self):
        # A tensor read from the environment becomes a graph placeholder whose
        # Dynamo source is not rooted at an argument; it would only exist in the
        # caller's module, so the artifact would serve a raw NameError. The
        # exact check names the source, whether the tensor is a bare global, a
        # class attribute, an attribute of a user module, or held on a slotted
        # instance (slot value or class attribute).
        for fn, source in (
            (_precompile_dynamo_global_tensor, "G['_GLOBAL_TENSOR']"),
            (
                _precompile_dynamo_class_attr_tensor,
                "G['_PrecompileDynamoTensorClassAttr'].tensor",
            ),
            (
                _precompile_dynamo_module_attr_tensor,
                "G['_DYNAMO_TENSOR_MODULE'].weight",
            ),
            # A class attribute read through a slotted instance renders via the
            # type's __dict__; the global is still named.
            (
                _precompile_dynamo_slotted_class_attr_tensor,
                "G['_DYNAMO_SLOTTED_CLASS_ATTR']",
            ),
            (_precompile_dynamo_slot_value_tensor, "G['_DYNAMO_SLOTTED_VALUE'].t"),
        ):
            with self.subTest(fn=fn.__name__):
                with self.assertRaisesRegex(
                    PrecompileError, "from the Python environment"
                ) as cm:
                    torch.compiler.precompile(
                        fn,
                        example_inputs=[(torch.randn(3),)],
                        tracer="dynamo",
                        backend="eager",
                    )
                self.assertIn(source, str(cm.exception))

    def test_tracer_dynamo_accepts_singleton_input_shared_with_environment(self):
        # dtypes, layouts, and memory formats are process-wide singletons with
        # value-based guards, so one passed as input while also living on a
        # referenced config class is not an aliasing hazard; both variants of
        # each must capture and serve their own branch.
        x = torch.randn(3)
        cases = (
            (_precompile_dynamo_dtype_branch, torch.float32, torch.float64),
            (
                _precompile_dynamo_format_branch,
                torch.channels_last,
                torch.contiguous_format,
            ),
            (_precompile_dynamo_layout_branch, torch.strided, torch.sparse_coo),
        )
        for fn, match, other in cases:
            with self.subTest(fn=fn.__name__):
                code, cache = torch.compiler.precompile(
                    fn,
                    example_inputs=[(x, match), (x, other)],
                    tracer="dynamo",
                    backend="eager",
                )
                loaded = torch.compiler.precompile.load(code, cache)
                self.assertEqual(loaded(x, match), x * 2)
                self.assertEqual(loaded(x, other), x * 3)

    def test_tracer_dynamo_enum_exemption_semantics(self):
        # An UNUSED enum argument may pass through even when the referenced
        # enum class holds it (the exemption's win); a USED enum argument
        # fails capture loudly on its unserializable identity guard -- never
        # via the misleading aliasing error, never silently.
        x = torch.randn(3)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_enum_passthrough,
            example_inputs=[(x, _PrecompileDynamoMode.A)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(x, _PrecompileDynamoMode.B), x * 1)
        with self.assertRaisesRegex(PrecompileError, "identity guard"):
            torch.compiler.precompile(
                _precompile_dynamo_enum_branch,
                example_inputs=[(x, _PrecompileDynamoMode.A)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_module_carried_in_argument_slot(self):
        # Slot values are instance state: an nn.Module carried in a slot is
        # rejected exactly like one carried in __dict__.
        box = _PrecompileDynamoSlottedModuleBox()
        with self.assertRaisesRegex(PrecompileError, "nn.Module arguments"):
            torch.compiler.precompile(
                _precompile_dynamo_slotted_box_call,
                example_inputs=[(torch.randn(2), box)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_tensor_in_slotted_default(self):
        with self.assertRaisesRegex(PrecompileError, "tensor-valued function"):
            torch.compiler.precompile(
                _precompile_dynamo_slotted_tensor_default,
                example_inputs=[(torch.randn(3),)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_jagged_layout_input(self):
        # NJT aliasing cannot be verified by the storage-overlap probe, and
        # Dynamo does not reject jagged inputs itself; both capture and the
        # loaded artifact must refuse them loudly.
        nt = torch.nested.nested_tensor(
            [torch.randn(2), torch.randn(3)], layout=torch.jagged
        )
        with self.assertRaisesRegex(PrecompileError, "cannot verify storage overlap"):
            torch.compiler.precompile(
                _precompile_dynamo_torch_sin,
                example_inputs=[(nt,)],
                tracer="dynamo",
                backend="eager",
            )
        # The loaded artifact runs the overlap check only when a captured graph
        # mutates an input (_DYNAMO_MUTATES_INPUTS), so capture a mutating fn.
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_inplace_step,
            example_inputs=[(torch.randn(3),)],
            tracer="dynamo",
            backend="eager",
        )
        self.assertIn("_DYNAMO_MUTATES_INPUTS = True", code)
        loaded = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "cannot verify storage overlap"):
            loaded(nt)

    def test_tracer_dynamo_sparse_input_gets_dynamo_diagnostics(self):
        # Sparse layouts skip the overlap probe so Dynamo's own clearer
        # rejection surfaces instead of "cannot verify storage overlap".
        sparse = torch.eye(3).to_sparse()
        with self.assertRaisesRegex(PrecompileError, "(?i)sparse"):
            torch.compiler.precompile(
                _precompile_dynamo_torch_sin,
                example_inputs=[(sparse,)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_accepts_argument_with_module_class_attribute(self):
        # The widened tensor-global walk must not leak into the per-argument
        # nn.Module check: an argument is rejected only for what it carries,
        # not for unrelated attributes of its type.
        x = torch.randn(4)
        box = _PrecompileDynamoBoxWithModuleClassAttr(3.0)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_box_scale,
            example_inputs=[(x, box)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        served = loaded(x, _PrecompileDynamoBoxWithModuleClassAttr(3.0))
        self.assertEqual(served, x * 3.0)

    def test_tracer_dynamo_rejects_global_mutation(self):
        # Capture runs against a copy of the fn's globals and the artifact against
        # its own namespace, so a global mutation would be silently discarded;
        # the transformed bytecode's STORE_GLOBAL is rejected at artifact build.
        with self.assertRaisesRegex(PrecompileError, "mutates the global"):
            torch.compiler.precompile(
                _precompile_dynamo_mutates_global,
                example_inputs=[(torch.randn(3),)],
                tracer="dynamo",
                backend="eager",
            )
        self.assertIsNone(_DYNAMO_MUTATED_GLOBAL)

    def test_tracer_dynamo_rejects_overlapping_storage_inputs(self):
        base = torch.randn(6)
        with self.assertRaisesRegex(PrecompileError, "share or overlap storage"):
            torch.compiler.precompile(
                _precompile_dynamo_aliasing,
                example_inputs=[(base[0:4], base[2:6])],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_serve_rejects_overlapping_storage_inputs(self):
        # The AOT StorageOverlap relation has no serialized form, so an artifact
        # captured on non-overlapping inputs must raise on overlapping runtime
        # views instead of silently computing the wrong thing (the fn mutates
        # its first argument).
        a = torch.randn(4)
        b = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_aliasing,
            example_inputs=[(a, b)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        a2, b2 = torch.randn(4), torch.randn(4)
        expected = _precompile_dynamo_aliasing(a2.clone(), b2)
        self.assertEqual(loaded(a2, b2), expected)
        base = torch.randn(6)
        with self.assertRaisesRegex(PrecompileError, "share or overlap storage"):
            loaded(base[0:4], base[2:6])

    def test_tracer_dynamo_environment_is_specialized_at_capture(self):
        # The programming-model contract: environment guards are minimized away,
        # so changing the environment after capture silently serves the
        # capture-time specialization. Pin that this is specialization, not a
        # stale read at serve time.
        x = torch.randn(4)
        original = _DYNAMO_ENV_SCALE
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_env_scale,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        # Change the environment global after capture; mock.patch.object restores it.
        with mock.patch.object(sys.modules[__name__], "_DYNAMO_ENV_SCALE", 100):
            loaded = torch.compiler.precompile.load(code, cache)
            self.assertEqual(loaded(x), x * original)

    def test_tracer_dynamo_rejects_tensor_default(self):
        with self.assertRaisesRegex(PrecompileError, "tensor-valued function defaults"):
            torch.compiler.precompile(
                _precompile_dynamo_tensor_default,
                example_inputs=[(torch.randn(3),)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_tensor_inside_object_default(self):
        # A picklable object default holding a tensor must be rejected like a direct
        # tensor default: pickling it would embed real tensor bytes in the artifact.
        with self.assertRaisesRegex(PrecompileError, "tensor-valued function defaults"):
            torch.compiler.precompile(
                _precompile_dynamo_object_tensor_default,
                example_inputs=[(torch.randn(3),)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_unpicklable_default(self):
        with self.assertRaisesRegex(PrecompileError, "default values"):
            torch.compiler.precompile(
                _precompile_dynamo_unpicklable_default,
                example_inputs=[(torch.randn(3),)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_non_function_fn(self):
        with self.assertRaisesRegex(PrecompileError, "requires a Python function"):
            torch.compiler.precompile(
                torch.nn.Linear(2, 2),
                example_inputs=[(torch.randn(1, 2),)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_closure_fn(self):
        with self.assertRaisesRegex(PrecompileError, "closure cells"):
            torch.compiler.precompile(
                _make_precompile_dynamo_closure(),
                example_inputs=[(torch.randn(3),)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_module_example_argument(self):
        with self.assertRaisesRegex(PrecompileError, "nn.Module arguments"):
            torch.compiler.precompile(
                lambda m, x: m(x),
                example_inputs=[(torch.nn.Linear(2, 2), torch.randn(1, 2))],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_nested_module_example_argument(self):
        # The rejection walks containers: a module hidden inside an argument must
        # not slip into the (unsupported, untested) nested-module capture path.
        with self.assertRaisesRegex(PrecompileError, "nn.Module arguments"):
            torch.compiler.precompile(
                lambda pair: pair[0](pair[1]),
                example_inputs=[((torch.nn.Linear(2, 2), torch.randn(1, 2)),)],
                tracer="dynamo",
                backend="eager",
            )

    @unittest.skipUnless(TEST_NUMPY, "requires numpy")
    def test_tracer_dynamo_rejects_numpy_example_inputs(self):
        # Dynamo traces ndarrays via ___from_numpy sources whose TENSOR_MATCH
        # guard construction fails under the save-guards path, so capture
        # would die with an internal error; reject up front (bare and nested)
        # with actionable advice.
        import numpy as np

        arr = np.ones(3, dtype=np.float32)
        with self.assertRaisesRegex(PrecompileError, "torch.from_numpy"):
            torch.compiler.precompile(
                lambda a: a, example_inputs=[(arr,)], tracer="dynamo", backend="eager"
            )
        with self.assertRaisesRegex(PrecompileError, "torch.from_numpy"):
            torch.compiler.precompile(
                lambda t, box: t,
                example_inputs=[(torch.randn(3), [arr])],
                tracer="dynamo",
                backend="eager",
            )
        # numpy scalars (numpy.generic) route through the same ___from_numpy
        # guard path and died with a raw internal TypeError when only ndarray
        # was rejected.
        with self.assertRaisesRegex(PrecompileError, "torch.from_numpy"):
            torch.compiler.precompile(
                lambda s, t: t * s,
                example_inputs=[(np.float64(2.0), torch.randn(3))],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_decompositions(self):
        with self.assertRaisesRegex(
            PrecompileError, "decompositions are not yet supported"
        ):
            torch.compiler.precompile(
                _precompile_dynamo_torch_sin,
                example_inputs=[(torch.randn(3),)],
                tracer="dynamo",
                backend="eager",
                decompositions={},
            )

    def test_tracer_dynamo_rejects_callable_input_identity_guard(self):
        # A plain-function positional input produces a non-global identity guard
        # (CLOSURE_MATCH on L['cb']), which Dynamo cannot serialize; the capture
        # must surface it as PrecompileError NAMING the guard (the typed
        # GuardSerializationError flows through convert_frame's chained
        # PackageError), not a raw internal AssertionError. (A torch builtin
        # like torch.sin would NOT trigger this: trace_rules handles it
        # without an identity guard on the input.)
        with self.assertRaisesRegex(
            PrecompileError, r"CLOSURE_MATCH guard \(on L\['cb'\]\)"
        ):
            torch.compiler.precompile(
                _precompile_dynamo_callable_input,
                example_inputs=[(torch.randn(3), _precompile_dynamo_torch_sin)],
                tracer="dynamo",
                backend="eager",
            )

    def test_dynamo_eager_backend_rejects_get_attr(self):
        from torch._precompile import _build_dynamo_eager_graph_source

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("b", torch.ones(3))

            def forward(self, x):
                return x + self.b

        gm = torch.fx.symbolic_trace(M())
        with self.assertRaisesRegex(PrecompileError, "get_attr nodes"):
            _build_dynamo_eager_graph_source(gm)

    def test_tracer_dynamo_arity_mismatch_raises_precompile_error(self):
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_scalar,
            example_inputs=[(x, 2)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "missing a required argument"):
            loaded(x)
        # A dynamo artifact binds its arguments like the traced fn, so a keyword
        # call dispatches on the same guards as the positional one.
        self.assertEqual(loaded(x, scale=2), x + 2)
        with self.assertRaisesRegex(PrecompileError, "unexpected keyword"):
            loaded(x, scale=2, bogus=1)

    def test_load_dynamo_state_accessor(self):
        code, _ = torch.compiler.precompile(
            _precompile_dynamo_torch_sin,
            example_inputs=[(torch.randn(3),)],
            tracer="dynamo",
            backend="eager",
        )
        state = _load_dynamo_state(code)
        # Exact set: this pins the serialized format. Adding a key is fine but
        # must update this test (and be consumed somewhere) deliberately.
        self.assertEqual(
            set(state),
            {"code", "import_sources", "defaults", "kwdefaults", "variants"},
        )
        self.assertEqual(len(state["variants"]), 1)
        make_fx_code, _ = torch.compiler.precompile(
            lambda t: t + 1, example_inputs=[(torch.randn(3),)], backend="eager"
        )
        self.assertIsNone(_load_dynamo_state(make_fx_code))

    def test_tracer_dynamo_corrupt_cache_bundle_degrades_to_jit(self):
        # The cache is an acceleration only: a corrupt per-graph bundle inside the
        # dynamo list-of-bundles envelope must leave load() on the JIT path with
        # correct numerics. Byte corruption is swallowed (and warned about) inside
        # torch.compiler.load_cache_artifacts, not by precompile's own warning.
        x = torch.randn(3)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_torch_sin, example_inputs=[(x,)], tracer="dynamo"
        )
        blob = torch.load(io.BytesIO(cache), weights_only=True)  # pins weights_only
        self.assertIsInstance(blob["artifact"], list)
        # One real (non-empty) bundle per compiled graph: an inductor artifact
        # whose cache primes nothing would silently JIT everything at load.
        self.assertEqual(len(blob["artifact"]), 1)
        self.assertIsInstance(blob["artifact"][0], bytes)
        blob["artifact"] = [b"corrupt-bundle"]
        buf = io.BytesIO()
        torch.save(blob, buf)
        with self.assertLogs("torch.compiler._cache", level="WARNING"):
            loaded = torch.compiler.precompile.load(code, buf.getvalue())
        self.assertEqual(loaded(x), torch.sin(x))

    def test_tracer_dynamo_load_rejects_mismatched_code_cache_pair(self):
        # The in-memory pair hard-fails on a code_hash mismatch on both tracers
        # (invariant 7). Only load_files degrades: a stateful rewrite renames
        # the artifact then the cache, so a crash between them leaves exactly
        # this pair on disk and the "always loadable" contract has to cover it.
        x = torch.randn(3)
        code_a, _cache_a = torch.compiler.precompile(
            _precompile_dynamo_torch_sin, example_inputs=[(x,)], tracer="dynamo"
        )
        _code_b, cache_b = torch.compiler.precompile(
            _precompile_dynamo_dynamic, example_inputs=[(x,)], tracer="dynamo"
        )
        with self.assertRaisesRegex(PrecompileError, "code_hash"):
            torch.compiler.precompile.load(code_a, cache_b)
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = os.path.join(tmp, "f.py")
            cache_path = os.path.join(tmp, "f.cache")
            with open(artifact_path, "w") as f:
                f.write(code_a)
            with open(cache_path, "wb") as f:
                f.write(cache_b)
            with self.assertLogs("torch._precompile", level="WARNING") as logs:
                loaded = torch.compiler.precompile.load_files(artifact_path, cache_path)
        self.assertTrue(any("code_hash" in message for message in logs.output))
        self.assertEqual(loaded(x), torch.sin(x))

    def test_make_fx_load_rejects_mismatched_code_cache_pair(self):
        # make_fx artifacts keep the strict pairing check (invariant 7).
        x = torch.randn(3)
        code_a, _cache_a = torch.compiler.precompile(
            lambda t: t + 1, example_inputs=[(x,)], backend="eager"
        )
        _code_b, cache_b = torch.compiler.precompile(
            lambda t: t * 2, example_inputs=[(x,)], backend="eager"
        )
        with self.assertRaisesRegex(PrecompileError, "does not match python_code"):
            torch.compiler.precompile.load(code_a, cache_b)

    def test_tracer_dynamo_truncated_artifact_rejected_cleanly(self):
        # A dynamo artifact missing a required driver global (here the opaque
        # _DYNAMO_STATE section) must fail load's metadata check with a clean
        # PrecompileError, not a NameError from exec'ing the truncated source.
        x = torch.randn(3)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_torch_sin,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        lines = [
            line for line in code.splitlines() if not line.startswith("_DYNAMO_STATE =")
        ]
        with self.assertRaisesRegex(PrecompileError, "missing calling-convention"):
            torch.compiler.precompile.load("\n".join(lines), cache)

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_stateful_accumulates_and_rewrites(self):
        # Caller-owned loop: each call runs one example, returns its result plus
        # an opaque state, and rewrites a loadable artifact on disk. The second
        # call recompiles into a dynamic variant (one isolate bucket and one PGO
        # record ride in the state); the third is a pure guard hit and adds
        # nothing.
        with tempfile.TemporaryDirectory() as tmp:
            paths = _stateful_paths(tmp)
            state = None
            for shape, expected_variants in (((2, 4), 1), ((3, 4), 2), ((2, 4), 2)):
                x = torch.randn(*shape)
                [result], state = torch.compiler.precompile.stateful(
                    _precompile_dynamo_dynamic,
                    example_inputs=[(x,)],
                    state=state,
                    backend="eager",
                    **paths,
                )
                if state.calls == 1:
                    self.addCleanup(state.close)
                self.assertEqual(result, _precompile_dynamo_dynamic(x))
                code, cache = _read_pair(paths)
                self.assertIn(f"VARIANT_COUNT = {expected_variants}", code)
                loaded = torch.compiler.precompile.load(code, cache)
                self.assertEqual(loaded(x), _precompile_dynamo_dynamic(x))
            self.assertIn("DYNAMIC_GRAPH_COUNT = 1", code)
            summary = state.summary()
            self.assertEqual(
                (summary.calls, summary.examples, summary.variants),
                (3, 3, 2),
            )
            self.assertEqual(summary.dynamic_graphs, 1)
            loaded = torch.compiler.precompile.load_files(**paths)
            y = torch.randn(7, 4)
            self.assertEqual(loaded(y), _precompile_dynamo_dynamic(y))

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_stateful_training_backward_between_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _stateful_paths(tmp)
            state = None
            for shape in ((2, 3), (4, 5)):
                x = torch.randn(*shape, requires_grad=True)
                [out], state = torch.compiler.precompile.stateful(
                    _precompile_dynamo_dynamic,
                    example_inputs=[(x,)],
                    state=state,
                    **paths,
                )
                if state.calls == 1:
                    self.addCleanup(state.close)
                # Each call is a real training step: its backward runs before
                # the next call, exactly like a caller-owned loop would.
                out.sum().backward()
                self.assertEqual(x.grad, x.detach().cos())
            code, cache = _read_pair(paths)
            self.assertIn("DYNAMIC_GRAPH_COUNT = 1", code)
            loaded = torch.compiler.precompile.load(code, cache)
            y = torch.randn(7, 4, requires_grad=True)
            served = loaded(y)
            served.sum().backward()
            self.assertEqual(y.grad, y.detach().cos())

    def test_tracer_dynamo_drops_import_rooted_global_guards(self):
        # A pytree call installs an ID_MATCH guard rooted at ImportSource
        # (__import__('torch')): provenance GLOBAL, so capture must drop it
        # like its GlobalSource-rooted siblings instead of aborting when guard
        # serialization rejects it.
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_pytree_sum,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(x), _precompile_dynamo_pytree_sum(x))

    def test_tracer_dynamo_stateful_rejects_equal_paths(self):
        # Equal paths would make the cache write clobber the artifact temp
        # file mid-rewrite and destroy the previously loadable artifact.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "f.py")
            with self.assertRaisesRegex(
                ValueError, "distinct artifact_path and cache_path"
            ):
                torch.compiler.precompile.stateful(
                    _precompile_dynamo_torch_sin,
                    example_inputs=[(torch.randn(4),)],
                    state=None,
                    backend="eager",
                    artifact_path=path,
                    cache_path=path,
                )

    def test_tracer_dynamo_stateful_survives_a_call_that_raised(self):
        # A call that raises has already told the caller; later good calls must
        # keep capturing and rewriting instead of inheriting the failure.
        with tempfile.TemporaryDirectory() as tmp:
            paths = _stateful_paths(tmp)
            kwargs = {"backend": "eager", **paths}
            x = torch.randn(4)
            _, state = torch.compiler.precompile.stateful(
                _precompile_dynamo_stateful_flaky,
                example_inputs=[(x, 1)],
                state=None,
                **kwargs,
            )
            self.addCleanup(state.close)
            with self.assertRaisesRegex(PrecompileError, "graph breaks"):
                torch.compiler.precompile.stateful(
                    _precompile_dynamo_stateful_flaky,
                    example_inputs=[(x, 3)],
                    state=state,
                    **kwargs,
                )
            [result], state = torch.compiler.precompile.stateful(
                _precompile_dynamo_stateful_flaky,
                example_inputs=[(x, 2)],
                state=state,
                **kwargs,
            )
            self.assertEqual(result, x + 2)
            self.assertEqual(state.summary().calls, 2)  # the failed call added none
            loaded = torch.compiler.precompile.load_files(**paths)
            self.assertEqual(loaded(x, 1), x + 1)
            self.assertEqual(loaded(x, 2), x + 2)

    def test_tracer_dynamo_stateful_step_failure_after_recompile_keeps_capturing(self):
        # Dynamo installs a new guarded variant at frame entry, so a step that
        # raises AFTER compiling (a data-dependent runtime error) leaves a
        # variant behind. The example is recorded up front, so later calls must
        # keep capturing and rewriting rather than fail guard minimization with
        # an unmatched variant.
        with tempfile.TemporaryDirectory() as tmp:
            paths = _stateful_paths(tmp)
            kwargs = {"backend": "eager", "dynamic": False, **paths}
            ok = torch.tensor([0])
            _, state = torch.compiler.precompile.stateful(
                _precompile_dynamo_gather,
                example_inputs=[(torch.randn(2), ok)],
                state=None,
                **kwargs,
            )
            self.addCleanup(state.close)
            with self.assertRaisesRegex(IndexError, "out of bounds"):
                # New shape -> new static variant compiles, then the step fails.
                torch.compiler.precompile.stateful(
                    _precompile_dynamo_gather,
                    example_inputs=[(torch.randn(3), torch.tensor([99]))],
                    state=state,
                    **kwargs,
                )
            x = torch.randn(4)
            [result], state = torch.compiler.precompile.stateful(
                _precompile_dynamo_gather,
                example_inputs=[(x, ok)],
                state=state,
                **kwargs,
            )
            self.assertEqual(result, _precompile_dynamo_gather(x, ok))
            loaded = torch.compiler.precompile.load_files(**paths)
            self.assertEqual(loaded(x, ok), _precompile_dynamo_gather(x, ok))
            # The failed call's variant stayed captured: its shape serves (and
            # reproduces eager, including the failure mode, on bad data).
            y = torch.randn(3)
            self.assertEqual(loaded(y, ok), _precompile_dynamo_gather(y, ok))

    def test_tracer_dynamo_stateful_rejects_unbindable_example(self):
        # A wrong-arity example must raise without being recorded: recorded
        # examples are signature.bind()'d on every rebuild, so recording it
        # would poison every later call on the state.
        with tempfile.TemporaryDirectory() as tmp:
            paths = _stateful_paths(tmp)
            kwargs = {"backend": "eager", **paths}
            x = torch.randn(4)
            _, state = torch.compiler.precompile.stateful(
                _precompile_dynamo_scalar, example_inputs=[(x, 1)], state=None, **kwargs
            )
            self.addCleanup(state.close)
            with self.assertRaisesRegex(TypeError, "does not match the positional"):
                torch.compiler.precompile.stateful(
                    _precompile_dynamo_scalar,
                    example_inputs=[(x, 1, 2)],
                    state=state,
                    **kwargs,
                )
            # A batch with a bad example records NOTHING, including the good
            # examples before it -- the message says so.
            with self.assertRaisesRegex(TypeError, "No example from this call"):
                torch.compiler.precompile.stateful(
                    _precompile_dynamo_scalar,
                    example_inputs=[(x, 2), (x, 1, 2)],
                    state=state,
                    **kwargs,
                )
            [result], state = torch.compiler.precompile.stateful(
                _precompile_dynamo_scalar,
                example_inputs=[(x, 3)],
                state=state,
                **kwargs,
            )
            self.assertEqual(result, x + 3)
            # 1 from the fresh call + 1 from the good call: the two failed
            # calls contributed no examples.
            self.assertEqual(state.summary().examples, 2)
            loaded = torch.compiler.precompile.load_files(**paths)
            self.assertEqual(loaded(x, 3), x + 3)

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_stateful_summary_and_dropped_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _stateful_paths(tmp)
            state = None
            for size in (3, 4):  # the second call recompiles a dynamic variant
                _, state = torch.compiler.precompile.stateful(
                    _precompile_dynamo_env_scale,
                    example_inputs=[(torch.randn(size),)],
                    state=state,
                    backend="eager",
                    **paths,
                )
                if state.calls == 1:
                    self.addCleanup(state.close)
            summary = state.summary()
            self.assertEqual(summary.calls, 2)
            self.assertEqual(summary.examples, 2)
            self.assertEqual(summary.variants, 2)
            self.assertEqual(summary.graphs, 2)
            self.assertEqual(summary.dynamic_graphs, 1)
            # The fn reads a module-global constant, whose environment guards
            # are dropped from dispatch and reported -- with the capture-time
            # value the artifact is specialized to -- in the summary and in the
            # artifact itself. Reporting aggregates across variants and
            # deduplicates: the triple dropped from both variants appears once.
            entry = ("EQUALS_MATCH", "G['_DYNAMO_ENV_SCALE']", "3")
            self.assertIn(entry, summary.dropped_guards)
            self.assertEqual(
                len(summary.dropped_guards), len(set(summary.dropped_guards))
            )
            code, _cache = _read_pair(paths)
            self.assertIn("_DROPPED_GUARDS = (", code)
            self.assertIn("dropped from at least one", code)
            from torch._precompile import _parse_artifact_metadata

            embedded = _parse_artifact_metadata(code)["_DROPPED_GUARDS"]
            self.assertEqual(embedded, tuple(summary.dropped_guards))

    def test_tracer_dynamo_dynamic_kwarg(self):
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
            dynamic=True,
        )
        self.assertIn("DYNAMIC_GRAPH_COUNT = 1", code)
        loaded = torch.compiler.precompile.load(code, cache)
        y = torch.randn(9, 4)
        self.assertEqual(loaded(y), _precompile_dynamo_dynamic(y))

        with self.assertRaisesRegex(ValueError, "require tracer='dynamo'"):
            torch.compiler.precompile(
                lambda t: t + 1, example_inputs=[(x,)], dynamic=True
            )

        with tempfile.TemporaryDirectory() as tmp:
            paths = _stateful_paths(tmp)
            _, state = torch.compiler.precompile.stateful(
                _precompile_dynamo_dynamic,
                example_inputs=[(x,)],
                state=None,
                backend="eager",
                dynamic=False,
                **paths,
            )
            self.addCleanup(state.close)
            with self.assertRaisesRegex(ValueError, "dynamic=True"):
                torch.compiler.precompile.stateful(
                    _precompile_dynamo_dynamic,
                    example_inputs=[(x,)],
                    state=state,
                    backend="eager",
                    dynamic=True,
                    **paths,
                )

    def test_write_artifact_files_failure_windows(self):
        # Each file is an fsync'd write_atomic (tmp + rename) in artifact-then-
        # cache order. A failure writing the artifact leaves the previous pair; a
        # failure writing the cache leaves the NEW artifact with the OLD cache
        # (the window load_files degrades on). Neither leaves temp files behind.
        import pathlib

        from torch._precompile import _write_dynamo_artifact_files

        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = os.path.join(tmp, "a.py")
            cache_path = os.path.join(tmp, "a.cache")
            with mock.patch("os.fsync", wraps=os.fsync) as fsync:
                _write_dynamo_artifact_files(
                    "GOOD = 1\n", b"goodcache", artifact_path, cache_path
                )
            self.assertEqual(fsync.call_count, 2)  # both files are made durable

            real_open = pathlib.Path.open
            failures = {"remaining": 0}

            def flaky_open(self_path, *args, **kwargs):
                if failures["remaining"] == 0 and str(self_path).endswith(".tmp"):
                    raise OSError("disk full")
                failures["remaining"] -= 1
                return real_open(self_path, *args, **kwargs)

            for fail_at, expected in (
                (0, ("GOOD = 1\n", b"goodcache")),  # first write fails: old pair
                (1, ("NEW = 2\n", b"goodcache")),  # cache write fails: new + old
            ):
                with self.subTest(fail_at=fail_at):
                    failures["remaining"] = fail_at
                    with mock.patch.object(pathlib.Path, "open", flaky_open):
                        with self.assertRaisesRegex(OSError, "disk full"):
                            _write_dynamo_artifact_files(
                                "NEW = 2\n", b"newcache", artifact_path, cache_path
                            )
                    with open(artifact_path) as f:
                        self.assertEqual(f.read(), expected[0])
                    with open(cache_path, "rb") as f:
                        self.assertEqual(f.read(), expected[1])
                    self.assertEqual(
                        [f for f in os.listdir(tmp) if f.endswith(".tmp")], []
                    )

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_stateful_rewrite_interrupted_between_renames(self):
        # End-to-end atomic rewrite through a real stateful call:
        # _write_dynamo_artifact_files renames the artifact then the cache. A
        # crash before the FIRST rename leaves the previous good pair (still
        # loadable); a crash between the two leaves the NEW artifact with the OLD
        # cache, which load_files degrades on (code_hash mismatch -> cold cache
        # with a warning) and still serves. Simulate each crash by failing the
        # write_atomic call for the matching path.
        from torch._inductor import codecache

        real_write = codecache.write_atomic

        def fail_writing(target):
            def flaky(path, content, **kwargs):
                if os.path.realpath(str(path)) == os.path.realpath(target):
                    raise OSError("disk full")
                return real_write(path, content, **kwargs)

            return flaky

        fn = _precompile_dynamo_dynamic
        with tempfile.TemporaryDirectory() as tmp:
            paths = _stateful_paths(tmp)
            _, state = torch.compiler.precompile.stateful(
                fn,
                example_inputs=[(torch.randn(2, 4),)],
                state=None,
                backend="eager",
                **paths,
            )
            self.addCleanup(state.close)
            good_code, good_cache = _read_pair(paths)

            # Crash before the FIRST rename (artifact write fails): the previous
            # good pair is untouched and still loads/serves what it captured.
            with mock.patch.object(
                codecache, "write_atomic", fail_writing(paths["artifact_path"])
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    torch.compiler.precompile.stateful(
                        fn,
                        example_inputs=[(torch.randn(3, 4),)],
                        state=state,
                        backend="eager",
                        **paths,
                    )
            self.assertEqual(_read_pair(paths), (good_code, good_cache))
            x2 = torch.randn(2, 4)
            self.assertEqual(torch.compiler.precompile.load_files(**paths)(x2), fn(x2))

            # Crash between the two renames (cache write fails): the NEW artifact
            # lands (the size-3 variant added above), the OLD cache stays, and
            # load_files degrades on the mismatch but still serves.
            with mock.patch.object(
                codecache, "write_atomic", fail_writing(paths["cache_path"])
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    torch.compiler.precompile.stateful(
                        fn,
                        example_inputs=[(torch.randn(5, 4),)],
                        state=state,
                        backend="eager",
                        **paths,
                    )
            new_code, cache_now = _read_pair(paths)
            # The artifact was updated (its rename ran) but the cache is stale (old).
            self.assertNotEqual(new_code, good_code)
            self.assertEqual(cache_now, good_cache)
            with self.assertLogs("torch._precompile", level="WARNING") as logs:
                loaded = torch.compiler.precompile.load_files(**paths)
            self.assertTrue(
                any(
                    "does not match" in m or "rewrite interrupted" in m
                    for m in logs.output
                )
            )
            z = torch.randn(5, 4)
            self.assertEqual(loaded(z), fn(z))

    def test_load_argument_types_validated(self):
        with self.assertRaisesRegex(TypeError, "python_code must be a str"):
            torch.compiler.precompile.load(b"code", b"cache")
        with self.assertRaisesRegex(TypeError, "cache must be bytes or None"):
            torch.compiler.precompile.load("code", "cache")
        with self.assertRaisesRegex(TypeError, "artifact_path"):
            torch.compiler.precompile.load("code", b"cache", artifact_path="a.py")

    def test_load_from_paths_make_fx_artifact(self):
        # load_files is exercised elsewhere only on dynamo artifacts; a make_fx
        # pair written to disk must load via it too, and a MISSING artifact file
        # raises the documented FileNotFoundError (only a missing CACHE file
        # degrades -- see test_tracer_dynamo_load_degrades_on_missing_cache_file).
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            lambda t: t + 1, example_inputs=[(x,)], backend="eager"
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = os.path.join(tmp, "f.py")
            cache_path = os.path.join(tmp, "f.cache")
            with open(artifact_path, "w") as f:
                f.write(code)
            with open(cache_path, "wb") as f:
                f.write(cache)
            loaded = torch.compiler.precompile.load_files(artifact_path, cache_path)
            self.assertEqual(loaded(x), x + 1)
            with self.assertRaisesRegex(FileNotFoundError, "missing.py"):
                torch.compiler.precompile.load_files(
                    os.path.join(tmp, "missing.py"), cache_path
                )

    def test_tracer_dynamo_stateful_inplace_flip_keeps_state_healthy(self):
        # x.add_(y_requires_grad) flips the live x's requires_grad while the
        # recorded example runs; the state's snapshots must keep the entry
        # state or every later rebuild fails guard re-checking.
        def make():
            return torch.randn(4), torch.randn(4, requires_grad=True)

        with tempfile.TemporaryDirectory() as tmp:
            paths = _stateful_paths(tmp)
            _, state = torch.compiler.precompile.stateful(
                _precompile_dynamo_inplace_add,
                example_inputs=[make()],
                state=None,
                **paths,
            )
            self.addCleanup(state.close)
            # The rebuild on a later call re-checks the earlier (mutated)
            # recorded example; a poisoned snapshot fails here.
            _, state = torch.compiler.precompile.stateful(
                _precompile_dynamo_inplace_add,
                example_inputs=[make()],
                state=state,
                **paths,
            )
            code, cache = _read_pair(paths)
            loaded = torch.compiler.precompile.load(code, cache)
            p, q = make()
            loaded(p, q).backward()
            self.assertEqual(q.grad, torch.ones(4))

    def test_tracer_dynamo_stateful_close_releases_session(self):
        from torch._dynamo.utils import guard_failures

        with tempfile.TemporaryDirectory() as tmp:
            paths = _stateful_paths(tmp)
            state = None
            with torch._dynamo.config.patch(
                automatic_dynamic_shapes=True, assume_static_by_default=True
            ):
                for size in (2, 3):  # the recompile logs a guard failure
                    _, state = torch.compiler.precompile.stateful(
                        _precompile_dynamo_dynamic,
                        example_inputs=[(torch.randn(size, 4),)],
                        state=state,
                        backend="eager",
                        **paths,
                    )
                    if state.calls == 1:
                        # Idempotent; the explicit close() below is what is tested.
                        self.addCleanup(state.close)
            from torch._dynamo.eval_frame import cached_backends

            code_obj = state.capture_target.__code__
            self.assertIn(code_obj, guard_failures)  # the pin close() must release
            self.assertIn(id(state.backend_fn), cached_backends)
            state.close()
            state.close()  # idempotent
            self.assertNotIn(code_obj, guard_failures)
            self.assertNotIn(id(state.backend_fn), cached_backends)
            self.assertIn("closed", repr(state))
            with self.assertRaisesRegex(ValueError, "closed state"):
                torch.compiler.precompile.stateful(
                    _precompile_dynamo_dynamic,
                    example_inputs=[(torch.randn(2, 4),)],
                    state=state,
                    backend="eager",
                    **paths,
                )
            # The files written before close() remain a valid artifact.
            code_text, cache = _read_pair(paths)
            x = torch.randn(3, 4)
            loaded = torch.compiler.precompile.load(code_text, cache)
            self.assertEqual(loaded(x), _precompile_dynamo_dynamic(x))

    def test_tracer_dynamo_stateful_failed_mask_compile_raises_at_capture(self):
        # A tangent mask whose deferred compile fails is a capture-time error
        # naming the pattern -- the state can be closed and recaptured without
        # the offending example -- not a warning that drops the pattern until it
        # fails at serve time.
        from torch._functorch._aot_autograd.to_standalone_python import (
            _CompileToPythonState,
        )

        with tempfile.TemporaryDirectory() as tmp:
            paths = _stateful_paths(tmp)
            x = torch.randn(4, requires_grad=True)
            y = torch.randn(4, requires_grad=True)
            [(out_a, _)], state = torch.compiler.precompile.stateful(
                _precompile_dynamo_independent_outputs,
                example_inputs=[(x, y)],
                state=None,
                **paths,
            )
            self.addCleanup(state.close)
            original = _CompileToPythonState._compile_mask

            def boom(self, mask):
                if mask != 0:
                    raise RuntimeError("injected mask-compile failure")
                return original(self, mask)

            with mock.patch.object(_CompileToPythonState, "_compile_mask", boom):
                # The live hook records the partial pattern's mask before it
                # compiles, so this failed backward leaves the mask observed.
                with self.assertRaisesRegex(RuntimeError, "injected mask-compile"):
                    out_a.sum().backward()
                with self.assertRaisesRegex(
                    PrecompileError,
                    r"output-tangent pattern\(s\) \['0b10'\].*can no longer be rendered",
                ):
                    torch.compiler.precompile.stateful(
                        _precompile_dynamo_independent_outputs,
                        example_inputs=[(x.detach().clone().requires_grad_(), y)],
                        state=state,
                        **paths,
                    )
            # The last successfully written pair (mask 0 only) stays loadable.
            loaded = torch.compiler.precompile.load_files(**paths)
            p = torch.randn(4, requires_grad=True)
            q = torch.randn(4, requires_grad=True)
            served_a, served_b = loaded(p, q)
            (served_a.sum() + served_b.sum()).backward()
            self.assertEqual(p.grad, p.detach().cos())

    def test_tracer_dynamo_rejects_disabled_dynamo(self):
        with mock.patch.dict(os.environ, {"TORCHDYNAMO_DISABLE": "1"}):
            with self.assertRaisesRegex(PrecompileError, "disabled in this process"):
                torch.compiler.precompile(
                    _precompile_dynamo_torch_sin,
                    example_inputs=[(torch.randn(3),)],
                    tracer="dynamo",
                    backend="eager",
                )

    @torch._dynamo.config.patch(suppress_errors=True)
    def test_tracer_dynamo_capture_with_suppress_errors_enabled(self):
        # An ambient suppress_errors=True (e.g. TORCHDYNAMO_SUPPRESS_ERRORS=1)
        # must not break capture: eval_frame asserts suppress_errors is off
        # whenever fail_on_recompile_limit_hit is on, so the capture context
        # patches it False (an artifact must never be built from a silently
        # uncaptured eager fallback anyway). Both entry points must work.
        x = torch.randn(3)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_torch_sin,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        self.assertEqual(torch.compiler.precompile.load(code, cache)(x), torch.sin(x))
        with tempfile.TemporaryDirectory() as tmp:
            [result], state = torch.compiler.precompile.stateful(
                _precompile_dynamo_torch_sin,
                example_inputs=[(x,)],
                state=None,
                backend="eager",
                **_stateful_paths(tmp),
            )
            self.addCleanup(state.close)
            self.assertEqual(result, torch.sin(x))

    def test_tracer_dynamo_stateful_validation(self):
        x = torch.randn(3)
        # Both paths are mandatory keyword-only arguments of stateful().
        with self.assertRaisesRegex(TypeError, "cache_path"):
            torch.compiler.precompile.stateful(
                _precompile_dynamo_torch_sin,
                example_inputs=[(x,)],
                backend="eager",
                artifact_path="only-one.py",
            )
        with self.assertRaisesRegex(ValueError, "require tracer='dynamo'"):
            torch.compiler.precompile(
                lambda t: t + 1, example_inputs=[(x,)], recompile_limit=4
            )
        with tempfile.TemporaryDirectory() as tmp:
            paths = _stateful_paths(tmp)
            _, state = torch.compiler.precompile.stateful(
                _precompile_dynamo_torch_sin,
                example_inputs=[(x,)],
                backend="eager",
                **paths,
            )
            self.addCleanup(state.close)
            resume = {"example_inputs": [(x,)], "state": state}
            with self.assertRaisesRegex(ValueError, "mixed artifact"):
                torch.compiler.precompile.stateful(
                    _precompile_dynamo_torch_sin, backend="inductor", **resume, **paths
                )
            with self.assertRaisesRegex(ValueError, "resumes only the function"):
                torch.compiler.precompile.stateful(
                    _precompile_dynamo_dynamic, backend="eager", **resume, **paths
                )
            with self.assertRaisesRegex(ValueError, "recompile_limit"):
                torch.compiler.precompile.stateful(
                    _precompile_dynamo_torch_sin,
                    backend="eager",
                    recompile_limit=99,
                    **resume,
                    **paths,
                )
            with self.assertRaisesRegex(TypeError, "previous stateful precompile"):
                torch.compiler.precompile.stateful(
                    _precompile_dynamo_torch_sin,
                    example_inputs=[(x,)],
                    backend="eager",
                    state=object(),
                    **paths,
                )
            with self.assertRaisesRegex(ValueError, "grad mode"):
                with torch.no_grad():
                    torch.compiler.precompile.stateful(
                        _precompile_dynamo_torch_sin, backend="eager", **resume, **paths
                    )

    def test_tracer_dynamo_stateful_example_inputs_validation(self):
        # stateful() runs the same example_inputs checks as the one-shot entry
        # point (which test_example_inputs_require_tuples and
        # test_make_fx_requires_one_example_input cover): an empty batch and a
        # non-tuple example are rejected before anything is captured or
        # written.
        with tempfile.TemporaryDirectory() as tmp:
            paths = _stateful_paths(tmp)
            with self.assertRaisesRegex(ValueError, "at least one"):
                torch.compiler.precompile.stateful(
                    _precompile_dynamo_torch_sin,
                    example_inputs=[],
                    state=None,
                    backend="eager",
                    **paths,
                )
            with self.assertRaisesRegex(TypeError, "positional-argument tuples"):
                torch.compiler.precompile.stateful(
                    _precompile_dynamo_torch_sin,
                    example_inputs=[[torch.randn(3)]],
                    state=None,
                    backend="eager",
                    **paths,
                )
            self.assertEqual(os.listdir(tmp), [])  # nothing was written

    def test_tracer_dynamo_recompile_limit_kwarg(self):
        with torch._dynamo.config.patch(automatic_dynamic_shapes=False):
            examples = [(torch.randn(2, 4),), (torch.randn(3, 4),)]
            with self.assertRaisesRegex(PrecompileError, "recompile_limit=1"):
                torch.compiler.precompile(
                    _precompile_dynamo_dynamic,
                    example_inputs=examples,
                    tracer="dynamo",
                    backend="eager",
                    recompile_limit=1,
                )
            code, cache = torch.compiler.precompile(
                _precompile_dynamo_dynamic,
                example_inputs=examples,
                tracer="dynamo",
                backend="eager",
                recompile_limit=8,
            )
            loaded = torch.compiler.precompile.load(code, cache)
            x = torch.randn(3, 4)
            self.assertEqual(loaded(x), _precompile_dynamo_dynamic(x))

    def test_tracer_dynamo_example_with_populated_grad(self):
        # A live example tensor already carrying .grad (any real training loop)
        # must serialize: the guard pickler used to emit the grad as a _Missing
        # placeholder that load_guards_state could not assign back.
        x = torch.randn(3, requires_grad=True)
        x.sum().backward()
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_torch_sin,
            example_inputs=[(x,)],
            tracer="dynamo",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        y = torch.randn(3, requires_grad=True)
        self.assertEqual(loaded(y), torch.sin(y))

    # Crossref installs a torch-function mode during capture; the fresh
    # plain subprocess does not have it, so the artifact correctly rejects
    # the changed environment (the contract's declared invariant).
    @skipIfCrossRef
    @parametrize("mode", ("inference", "training"))
    def test_tracer_dynamo_source_runs_in_fresh_process(self, mode):
        # The self-contained artifact must exec and run in a fresh process with
        # the compiler hard-disabled and EMPTY on-disk kernel caches (its kernels
        # JIT from the inlined source; a serve path that needed Dynamo would
        # crash here -- the no-frame-compile property itself is pinned by
        # test_tracer_dynamo_load_runs_in_fresh_process). The training variant
        # additionally runs a real backward in the child and checks the grad.
        training = mode == "training"
        fn = _precompile_dynamo_dynamic if training else _precompile_dynamo_torch_sin
        code, _cache = torch.compiler.precompile(
            fn,
            example_inputs=[(torch.randn(4, requires_grad=training),)],
            tracer="dynamo",
        )
        check = (
            "x = t.randn(4, requires_grad=True)\n"
            "out = namespace['forward'](x)\n"
            "if not out.requires_grad:\n"
            "    raise AssertionError('expected a differentiable output')\n"
            "out.sum().backward()\n"
            "t.testing.assert_close(x.grad, x.detach().cos())\n"
            if training
            else "x = t.randn(4)\n"
            "t.testing.assert_close(namespace['forward'](x), t.sin(x))\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = os.path.join(tmp, "f.py")
            with open(artifact_path, "w") as f:
                f.write(code)
            driver = (
                "import runpy as r\n"
                "import sys as s\n"
                "import torch as t\n"
                "namespace = r.run_path(s.argv[1])\n"
            ) + check
            subprocess.check_call(
                [sys.executable, "-c", driver, artifact_path],
                env=_fresh_cache_env(tmp, TORCHDYNAMO_DISABLE="1"),
                timeout=300,
            )

    # Crossref installs a torch-function mode during capture; the fresh
    # plain subprocess does not have it, so the artifact correctly rejects
    # the changed environment (the contract's declared invariant).
    @skipIfCrossRef
    def test_tracer_dynamo_load_runs_in_fresh_process(self):
        # The public load_files() path, cold: a fresh subprocess with EMPTY
        # inductor and Triton on-disk caches has no warm state to mask a broken
        # artifact or an empty bundle. Two observables: load_cache_artifacts must
        # receive the bundle and report a populated FxGraph ("inductor") entry
        # (the bundle used to be [None]), and Dynamo -- enabled in the child --
        # must compile no frame. Kernel compilation is deliberately NOT asserted:
        # the inlined artifact runs its kernels directly, without a compile_fx
        # re-entry, so the FxGraph entry the bundle carries cannot skip the
        # g++/Triton JIT in an empty cache dir (only warm on-disk kernel caches do).
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_torch_sin, example_inputs=[(x,)], tracer="dynamo"
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = os.path.join(tmp, "f.py")
            cache_path = os.path.join(tmp, "f.cache")
            with open(artifact_path, "w") as f:
                f.write(code)
            with open(cache_path, "wb") as f:
                f.write(cache)
            subprocess.check_call(
                [
                    sys.executable,
                    "-c",
                    textwrap.dedent(
                        """
                        import sys as s
                        import torch as t

                        primed = []
                        original = t.compiler.load_cache_artifacts

                        def spy(bundle):
                            info = original(bundle)
                            primed.append(None if info is None else dict(info.artifacts))
                            return info

                        t.compiler.load_cache_artifacts = spy
                        loaded = t.compiler.precompile.load_files(s.argv[1], s.argv[2])
                        x = t.randn(4)
                        t.testing.assert_close(loaded(x), t.sin(x))
                        if len(primed) != 1 or not primed[0] or not primed[0].get("inductor"):
                            raise AssertionError(f"bundle was not consumed: {primed}")
                        # Serving must never compile a frame: dynamo is enabled
                        # in this child, so any compile would show up here.
                        import os as o

                        if o.environ.get("TORCHDYNAMO_DISABLE"):
                            raise AssertionError("dynamo must be enabled here")
                        from torch._dynamo.utils import counters

                        if counters["frames"]:
                            raise AssertionError(dict(counters["frames"]))
                        """
                    ),
                    artifact_path,
                    cache_path,
                ],
                # An ambient TORCHDYNAMO_DISABLE would silently void the
                # frame-counter assertion; _fresh_cache_env strips it (the child
                # double-checks).
                env=_fresh_cache_env(tmp),
                timeout=300,
            )

    def test_tracer_dynamo_varargs_dispatch(self):
        x = torch.randn(4)
        y = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_varargs,
            example_inputs=[(x, y)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(x, y), x + y)

    @torch._dynamo.config.patch(recompile_limit=8)
    def test_tracer_dynamo_captures_more_than_default_recompile_limit(self):
        # dynamic=False keeps every example a distinct static variant (automatic
        # dynamic would fold them into one symbolic graph after two examples and
        # never approach the limit), so 9 variants genuinely require the capture
        # limit to be raised past the patched config default of 8.
        x = torch.randn(4)
        examples = [(x, value) for value in range(9)]
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_scalar,
            example_inputs=examples,
            tracer="dynamo",
            backend="eager",
            dynamic=False,
        )
        self.assertIn("VARIANT_COUNT = 9", code)
        loaded = torch.compiler.precompile.load(code, cache)
        for args in examples:
            self.assertEqual(loaded(*args), _precompile_dynamo_scalar(*args))

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_capture_does_not_leak_auto_dynamic_state(self):
        dynamic_code, _ = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(torch.randn(2, 4),), (torch.randn(3, 4),)],
            tracer="dynamo",
            backend="eager",
        )
        self.assertIn("DYNAMIC_GRAPH_COUNT = 1", dynamic_code)

        static_code, static_cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(torch.randn(5, 4),)],
            tracer="dynamo",
            backend="eager",
        )
        self.assertIn("DYNAMIC_GRAPH_COUNT = 0", static_code)
        x = torch.randn(5, 4)
        self.assertEqual(
            torch.compiler.precompile.load(static_code, static_cache)(x),
            _precompile_dynamo_dynamic(x),
        )

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_capture_does_not_leak_session(self):
        # A one-shot capture closes its state internally; teardown must release
        # the capture code object from Dynamo's recompile-logging registry
        # (guard_failures) and the backend from cached_backends -- otherwise the
        # whole session (including the copied fn globals) stays pinned until
        # torch._dynamo.reset(). A recompiling capture actually populates
        # guard_failures during capture, so a skipped teardown would leave the
        # code object behind here (verified). The vacuous predecessor only
        # checked the copied-globals dict, which capture throws away regardless
        # of teardown; the stateful close() path is covered by
        # test_tracer_dynamo_stateful_close_releases_session.
        from torch._dynamo.eval_frame import cached_backends
        from torch._dynamo.utils import guard_failures
        from torch._precompile import _new_dynamo_state

        captured = {}
        real_new_state = _new_dynamo_state

        def spy(*args, **kwargs):
            state = real_new_state(*args, **kwargs)
            captured["state"] = state
            return state

        with mock.patch("torch._precompile._new_dynamo_state", spy):
            torch.compiler.precompile(
                _precompile_dynamo_dynamic,
                # Two sizes recompile, so guard_failures is populated during capture.
                example_inputs=[(torch.randn(2, 4),), (torch.randn(3, 4),)],
                tracer="dynamo",
                backend="eager",
            )
        state = captured["state"]
        self.assertNotIn(state.capture_target.__code__, guard_failures)
        self.assertNotIn(id(state.backend_fn), cached_backends)
        # The copy isolation also means capture never installs its generated
        # globals into this real module.
        module_globals = sys.modules[__name__].__dict__
        leaked = [
            name
            for name in module_globals
            if name.startswith(("__compiled_fn_", "__builtins_dict___", "__import_"))
        ]
        self.assertEqual(leaked, [])

    def test_tracer_dynamo_load_does_not_copy_unrelated_module_globals(self):
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(torch.randn(4),)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertNotIn("_GLOBAL_TENSOR", loaded._loaded_forward.__globals__)

    def test_tracer_dynamo_python_minor_mismatch_uses_public_error(self):
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(torch.randn(4),)],
            tracer="dynamo",
            backend="eager",
        )
        version = tuple(sys.version_info[:2])
        incompatible = code.replace(
            f"_DYNAMO_PYTHON_VERSION = {version!r}",
            "_DYNAMO_PYTHON_VERSION = (0, 0)",
        )
        # The edited code no longer pairs with the cache (code_hash), so load
        # it without one; the version check in the exec'd driver raises the
        # public error.
        with self.assertRaisesRegex(PrecompileError, "produced on Python"):
            torch.compiler.precompile.load(incompatible, None)

    def test_tracer_dynamo_torch_version_mismatch_uses_public_error(self):
        # _DYNAMO_STATE pickles Dynamo internals with no cross-version story, so
        # a foreign torch build must fail with this message rather than an
        # arbitrary unpickling error.
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(torch.randn(4),)],
            tracer="dynamo",
            backend="eager",
        )
        incompatible = code.replace(
            f"_DYNAMO_TORCH_VERSION = {torch.__version__!r}",
            "_DYNAMO_TORCH_VERSION = '0.0.0'",
        )
        self.assertNotEqual(incompatible, code)
        with self.assertRaisesRegex(PrecompileError, "produced by torch"):
            torch.compiler.precompile.load(incompatible, None)

    def test_tracer_dynamo_version_check_precedes_package_use(self):
        # On a foreign build a moved/renamed loader symbol would raise ImportError;
        # the driver must run its version check FIRST so the user gets the
        # actionable version message instead. Spy on the package loader the driver
        # imports: a version-mismatched load must raise the version PrecompileError
        # WITHOUT ever reaching the loader, while a matching load does reach it.
        import torch._dynamo.package as package

        code, cache = torch.compiler.precompile(
            _precompile_dynamo_torch_sin,
            example_inputs=[(torch.randn(3),)],
            tracer="dynamo",
            backend="eager",
        )
        reached = []
        real_load = package.load_guards_state

        def spy(*args, **kwargs):
            reached.append(True)
            return real_load(*args, **kwargs)

        incompatible = code.replace(
            f"_DYNAMO_TORCH_VERSION = {torch.__version__!r}",
            "_DYNAMO_TORCH_VERSION = '0.0.0'",
        )
        self.assertNotEqual(incompatible, code)
        with mock.patch.object(package, "load_guards_state", spy):
            with self.assertRaisesRegex(PrecompileError, "produced by torch"):
                torch.compiler.precompile.load(incompatible, None)
        self.assertEqual(reached, [])  # version check ran before touching the loader
        # Sanity: a version-matching load DOES reach the package loader.
        with mock.patch.object(package, "load_guards_state", spy):
            torch.compiler.precompile.load(code, cache)
        self.assertTrue(reached)

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_training_recompiles_to_dynamic_graph(self):
        examples = [
            (torch.randn(rows, columns, requires_grad=True),)
            for rows, columns in ((2, 3), (4, 5), (6, 7))
        ]
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=examples,
            tracer="dynamo",
        )

        self.assertIn("DYNAMIC_GRAPH_COUNT = 1", code)
        self.assertIn("class _CompiledFunction(torch.autograd.Function):", code)
        self.assertIn("_inner_call_fw", code)
        self.assertIn("_inner_call_bw", code)
        # Serving/backward numerics live in the device-generic
        # TestPrecompileNumerics.test_tracer_dynamo_training_numerics.
        loaded = torch.compiler.precompile.load(code, cache)
        x = torch.randn(8, 9, requires_grad=True)
        self.assertTrue(loaded(x).requires_grad)

    def test_tracer_dynamo_differentiability_inferred_per_graph(self):
        # Differentiability mirrors torch.compile on the inductor backend: a
        # captured graph whose inputs require grad compiles as a joint
        # forward+backward (the served output carries grad_fn and backward
        # matches eager); a graph whose inputs do not require grad stays an
        # inference graph. requires_grad is guarded per input, so a serve call
        # with it flipped misses dispatch loudly.
        x = torch.randn(4, requires_grad=True)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_torch_sin, example_inputs=[(x,)], tracer="dynamo"
        )
        loaded = torch.compiler.precompile.load(code, cache)
        y = torch.randn(4, requires_grad=True)
        out = loaded(y)
        self.assertIsNotNone(out.grad_fn)
        out.sum().backward()
        self.assertEqual(y.grad, y.detach().cos())
        with self.assertRaisesRegex(PrecompileError, "no captured Dynamo variant"):
            loaded(torch.randn(4))  # requires_grad flipped off

        plain = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_torch_sin, example_inputs=[(plain,)], tracer="dynamo"
        )
        self.assertIsNone(torch.compiler.precompile.load(code, cache)(plain).grad_fn)

    def test_dynamo_backend_source_literal_roundtrip(self):
        source = 'slash = "\\\\n"\ntriple = \'"""\'\n'
        namespace = {}
        exec(
            compile(
                f"sources = (\n{_dynamo_backend_source_literal(source)}\n)",
                "<backend-sources>",
                "exec",
            ),
            namespace,
        )
        self.assertEqual(namespace["sources"], (source,))

    def test_tracer_dynamo_rejects_graph_break(self):
        with self.assertRaisesRegex(
            PrecompileError, "does not support graph breaks yet"
        ):
            torch.compiler.precompile(
                _precompile_dynamo_graph_break,
                example_inputs=[(torch.randn(4),)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_retains_input_scalar_guards(self):
        examples = [(torch.randn(4), scale) for scale in (2, 3)]
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_scalar,
            example_inputs=examples,
            tracer="dynamo",
            backend="eager",
        )

        loaded = torch.compiler.precompile.load(code, cache)
        summaries = _dynamo_serialized_guard_summary(code)
        self.assertTrue(
            any("CONSTANT_MATCH" in guards for guards, _, _, _ in summaries)
        )
        x = torch.randn(4)
        self.assertEqual(loaded(x, 2), _precompile_dynamo_scalar(x, 2))
        self.assertEqual(loaded(x, 4), _precompile_dynamo_scalar(x, 4))

    def test_tracer_dynamo_retains_call_wrapped_input_guards(self):
        # Guard minimization classifies input guards by their originating
        # source's root, not by an "L['name']" string prefix: a tuple-iterator
        # input's guards render as ___tuple_iterator_getitem(L['it'], i), which
        # the prefix test misclassified as environment and dropped -- the
        # artifact then silently served the capture-time values for any
        # iterator.
        from torch._precompile import _parse_artifact_metadata

        t = torch.randn(3)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_iter_sum,
            example_inputs=[(iter((2.0, 3.0)), t)],
            tracer="dynamo",
            backend="eager",
        )
        dropped = _parse_artifact_metadata(code)["_DROPPED_GUARDS"]
        self.assertFalse(
            any("tuple_iterator" in source for _, source, _ in dropped), dropped
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(iter((2.0, 3.0)), t), t + 5.0)
        # Different iterator values must MISS dispatch, not serve stale 5.0.
        with self.assertRaisesRegex(PrecompileError, "no captured Dynamo variant"):
            loaded(iter((5.0, 9.0)), t)

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_dispatch_serves_newest_variant_first(self):
        # The driver serves the first variant whose guards pass, and variants
        # are stored newest-first (matching live Dynamo's LRU-front-first
        # checks): a size matching BOTH the early static variant and the later
        # dynamic one must be served by the dynamic one. Both variants compute
        # the same value, so the numerics alone cannot tell the order apart;
        # spy on the LIVE driver's guard-manager check() calls instead. An
        # oldest-first driver would check the static variant first for size 2
        # and this test would fail.
        from torch._dynamo import package

        examples = [(torch.randn(size),) for size in (2, 4, 2)]
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_affine,
            example_inputs=examples,
            tracer="dynamo",
            backend="eager",
        )
        self.assertIn("VARIANT_COUNT = 2", code)

        # Wrap every guard manager the driver builds so each check() records the
        # variant's construction index (0 = first built = first served) and its
        # result, without disturbing dispatch.
        check_log: list[tuple[int, bool]] = []
        built = []
        real_load = package.load_guard_manager

        class _SpyManager:
            def __init__(self, inner, index):
                self._inner = inner
                self._index = index

            def check(self, scope):
                result = self._inner.check(scope)
                check_log.append((self._index, result))
                return result

            def __getattr__(self, name):
                return getattr(self._inner, name)

        def spy_load(*args, **kwargs):
            manager = _SpyManager(real_load(*args, **kwargs), len(built))
            built.append(manager)
            return manager

        with mock.patch.object(package, "load_guard_manager", spy_load):
            loaded = torch.compiler.precompile.load(code, cache)

        # Size 5 matches ONLY the dynamic variant, so the index that returns True
        # identifies it. Newest-first stores the dynamic variant at index 0.
        x5 = torch.randn(5)
        check_log.clear()
        self.assertEqual(loaded(x5), _precompile_dynamo_affine(x5))
        dynamic_index = next(index for index, ok in check_log if ok)
        self.assertEqual(dynamic_index, 0)  # dynamic variant served first
        # Size 2 matches BOTH variants; newest-first checks the dynamic variant
        # first and short-circuits, so ONLY the dynamic index is checked. An
        # oldest-first driver would check (and serve) the static variant here.
        x2 = torch.randn(2)
        check_log.clear()
        self.assertEqual(loaded(x2), _precompile_dynamo_affine(x2))
        self.assertEqual(check_log, [(dynamic_index, True)])

    def test_tracer_dynamo_preserves_relational_guards(self):
        shared = torch.ones(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_aliasing,
            example_inputs=[
                (torch.ones(4), torch.full((4,), 2.0)),
                (shared, shared),
            ],
            tracer="dynamo",
        )

        summaries = _dynamo_serialized_guard_summary(code)
        # The aliased variant must carry its relational guard through
        # serialization and minimization; TENSOR_MATCH alone would be true of
        # any two-tensor fn.
        self.assertTrue(any("DUPLICATE_INPUT" in types for types, _, _, _ in summaries))
        self.assertTrue(all(has_shape for _, _, _, has_shape in summaries))
        for _, loaded in _default_and_inlined_loaders(code, cache, "inductor"):
            expected_a = torch.ones(4)
            expected = _precompile_dynamo_aliasing(expected_a, torch.full((4,), 2.0))
            actual_a = torch.ones(4)
            actual = loaded(actual_a, torch.full((4,), 2.0))
            self.assertEqual(actual, expected)
            self.assertEqual(actual_a, expected_a)

            expected_shared = torch.ones(4)
            expected = _precompile_dynamo_aliasing(expected_shared, expected_shared)
            actual_shared = torch.ones(4)
            actual = loaded(actual_shared, actual_shared)
            self.assertEqual(actual, expected)
            self.assertEqual(actual_shared, expected_shared)

    def test_tracer_dynamo_preserves_key_order_guard_dependencies(self):
        forward = {"a": 2.0, "b": 3.0}
        reverse = {"b": 3.0, "a": 2.0}
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dict_order,
            example_inputs=[(torch.ones(4), forward), (torch.ones(4), reverse)],
            tracer="dynamo",
            backend="eager",
        )

        summaries = _dynamo_serialized_guard_summary(code)
        self.assertTrue(any(key_order for _, _, key_order, _ in summaries))
        loaded = torch.compiler.precompile.load(code, cache)
        x = torch.ones(4)
        self.assertEqual(loaded(x, forward), _precompile_dynamo_dict_order(x, forward))
        self.assertEqual(loaded(x, reverse), _precompile_dynamo_dict_order(x, reverse))

    def test_tracer_invalid_raises(self):
        a, b = torch.randn(4, 4), torch.randn(4, 4)
        with self.assertRaisesRegex(ValueError, "tracer must be 'make_fx' or 'dynamo'"):
            torch.compiler.precompile(
                lambda x, y: x + y, example_inputs=[(a, b)], tracer="nope"
            )

    def test_backend_default_is_inductor(self):
        # The default lowers through Inductor: the generated code inlines the Inductor
        # output module. Use a graph_partition-agnostic marker (the ``call = runner.call``
        # form is only emitted when config.graph_partition is on, which is off in fbcode).
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, _ = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        self.assertIn("Inductor output code", code)

    def test_inductor_graph_partition_off(self):
        # graph_partition defaults off in fbcode; the Inductor output module then exposes
        # a top-level ``def call(args):`` instead of ``call = runner.call``. The source
        # extractor must still find it (regression: it previously matched only the
        # runner.call form, so torch.compiler.precompile crashed in fbcode).
        import torch._inductor.config as ind_config

        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        with ind_config.patch(graph_partition=False):
            code, cache = torch.compiler.precompile(
                lambda model, xx: model(xx), example_inputs=[(m, x)]
            )
            self.assertNotIn("call = runner.call", code)  # non-partition form
            f_c = torch.compiler.precompile.load(code, cache)
            self.assertEqual(f_c(m, x), m(x))

    @parametrize("cache_config", ("force_disable_caches", "fx_graph_cache"))
    def test_inductor_caches_disabled(self, cache_config):
        # Source is captured off codegen (GraphLowering.save_output_code), not the cache
        # bundle, so precompile must work even when caching is disabled -- producing a
        # runnable python_code with an empty cache, not a misleading "non-cacheable HOP"
        # error. Covers force_disable_caches=True and fx_graph_cache=False (the
        # disabling value differs per knob).
        import torch._inductor.config as ind_config

        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        with ind_config.patch(**{cache_config: cache_config == "force_disable_caches"}):
            code, cache = torch.compiler.precompile(
                lambda model, xx: model(xx), example_inputs=[(m, x)]
            )
            # No saveable artifact when caches are off; the cache is empty.
            blob = torch.load(io.BytesIO(cache), weights_only=True)
            self.assertIsNone(blob["artifact"])
            # python_code still runs standalone (JITs from inlined source).
            ns = {"__name__": "_a"}
            exec(compile(code, "<a>", "exec"), ns)
            self.assertEqual(ns["forward"](m, x), m(x))
            # ...and load() falls back to the inlined path.
            self.assertEqual(torch.compiler.precompile.load(code, cache)(m, x), m(x))

    def test_inductor_cpp_wrapper_pinned_off(self):
        # cpp_wrapper would make Inductor emit a C++ ``call`` (no python module); a
        # python artifact cannot come from it, so compile_to_python pins it off. With
        # cpp_wrapper=True ambient, precompile must still produce a working python artifact.
        import torch._inductor.config as ind_config

        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        with ind_config.patch(cpp_wrapper=True):
            code, cache = torch.compiler.precompile(
                lambda model, xx: model(xx), example_inputs=[(m, x)]
            )
            f_c = torch.compiler.precompile.load(code, cache)
            self.assertEqual(f_c(m, x), m(x))

    def test_example_grad_restored_when_fn_raises(self):
        # If fn runs a backward then raises during the make_fx trace, the example
        # model's .grad must be restored (the snapshot/restore is in a finally), not
        # left clobbered -- precompile does not mutate the example model's grads.
        torch.manual_seed(0)
        m = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        for p in m.parameters():
            self.assertIsNone(p.grad)

        def boom(model, xx):
            model(xx).sum().backward()  # populates .grad on the lifted example params
            raise ValueError("boom")

        with self.assertRaisesRegex(ValueError, "boom"):
            torch.compiler.precompile(boom, example_inputs=[(m, x)])
        for n, p in m.named_parameters():
            self.assertIsNone(p.grad, f"{n}: example .grad must be restored on failure")

    def test_unbacked_capture_with_preexisting_grad(self):
        # Regression: in the mark_unbacked path the example params are fakeified BEFORE
        # the grad clear. A model with a pre-existing .grad (the warmup-step-then-
        # precompile flow) plus a backward in fn must still capture -- the clear must
        # precede fakeify so the fakes inherit no grad -- and the real .grad is restored.
        from torch._dynamo.decorators import mark_unbacked

        torch.manual_seed(0)
        m = torch.nn.Linear(4, 3)
        x = torch.randn(8, 4)
        m(x).sum().backward()  # warmup: populate .grad before precompile
        saved = {n: p.grad.clone() for n, p in m.named_parameters()}
        mark_unbacked(x, 0)
        code, _ = torch.compiler.precompile(
            lambda mm, t: mm(t).sum().backward(), example_inputs=[(m, x)]
        )
        self.assertIn("USER_INPUT_SHAPES = [(None, 4)]", code)  # dim 0 is dynamic
        for n, p in m.named_parameters():
            self.assertEqual(p.grad, saved[n])  # warmup grad restored, not clobbered

    def test_backend_eager_no_inductor_lowering(self):
        # backend="eager" skips Inductor: the generated code has no inductor ``call``
        # entry point, and instead embeds the readable captured ATen graph and the
        # eager driver. The eager backend has no kernels to accelerate, so the cache
        # is empty -- python_code is the whole artifact.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)], backend="eager"
        )
        self.assertIn('backend="eager"', code)
        self.assertNotIn("call = runner.call", code)
        self.assertIn("torch.ops.aten", code)  # readable captured graph

        # The cache holds no artifact (eager caches nothing); the backend tag lives in
        # python_code (the single source of truth). The envelope still carries the
        # integrity tag, with backend='eager' to match python_code.
        self.assertIn("BACKEND = 'eager'", code)
        from torch._precompile import _CACHE_FORMAT, _CACHE_VERSION

        blob = torch.load(io.BytesIO(cache), weights_only=False)
        self.assertEqual(
            set(blob), {"artifact", "format", "version", "backend", "code_hash"}
        )
        self.assertIsNone(blob["artifact"])  # eager has no compiled blob to bundle
        self.assertEqual(blob["format"], _CACHE_FORMAT)
        self.assertEqual(blob["version"], _CACHE_VERSION)
        self.assertEqual(blob["backend"], "eager")

    def test_backend_eager_self_contained_exec(self):
        # The eager python_code execs standalone with NO cache (the captured graph
        # is inlined) and runs, matching eager.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()
        x = torch.randn(5, 4)
        code, _cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)], backend="eager"
        )

        ns = {"__name__": "_eager"}
        exec(compile(code, "<eager>", "exec"), ns)
        self.assertEqual(ns["forward"](m, x), m(x))

    def test_preexisting_param_grad_capture_succeeds(self):
        # Precompiling a backward fn on a model whose params already carry a .grad (the
        # common warmup-step-then-precompile flow) must capture cleanly: the pre-existing
        # grad must be cleared before tracing, not baked as a constant (invariant 1).
        # Eager simply accumulates a second backward, so precompile must too.
        torch.manual_seed(0)
        m = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        m(x).sum().backward()  # warmup: params now carry a .grad
        self.assertIsNotNone(m.weight.grad)
        grad_before = m.weight.grad.clone()

        code, cache = torch.compiler.precompile(
            lambda model, xx: model(xx).sum().backward(), example_inputs=[(m, x)]
        )
        # Capture must not mutate the example model's pre-existing grad (restored).
        self.assertEqual(m.weight.grad, grad_before)

        run = torch.nn.Linear(4, 3)
        run.load_state_dict(m.state_dict())
        torch.compiler.precompile.load(code, cache)(run, x)  # run.grad starts None
        ref = torch.nn.Linear(4, 3)
        ref.load_state_dict(m.state_dict())
        ref(x).sum().backward()
        for (n, p), (_, rp) in zip(run.named_parameters(), ref.named_parameters()):
            self.assertEqual(p.grad, rp.grad, n)

    def test_nontensor_output_inductor_clean_error(self):
        # A non-tensor python value (float, complex, str, ...) in fn's output trips the
        # inductor backend's codegen assert; surface a clear PrecompileError (not a raw
        # InductorError) pointing to backend="eager". int / None outputs lower fine, and
        # eager handles the non-tensor value.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(2, 4)
        for bad in (3.14, 2 + 3j, "hi"):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(PrecompileError, "non-tensor Python value"):
                    torch.compiler.precompile(
                        lambda model, t, b=bad: (model(t), b), example_inputs=[(m, x)]
                    )
        for extra in (7, None):
            with self.subTest(extra=extra):
                code, cache = torch.compiler.precompile(
                    lambda model, t, e=extra: (model(t), e), example_inputs=[(m, x)]
                )
                self.assertEqual(
                    torch.compiler.precompile.load(code, cache)(m, x)[1], extra
                )
        ecode, ecache = torch.compiler.precompile(
            lambda model, t: (model(t), 3.14), example_inputs=[(m, x)], backend="eager"
        )
        self.assertEqual(torch.compiler.precompile.load(ecode, ecache)(m, x)[1], 3.14)

    def test_input_layout_mismatch_inductor_clean_error(self):
        # The inductor backend bakes each input's stride / memory format (invariant 6);
        # a same-shape input with a different layout must raise a clear PrecompileError
        # (not a raw assert_size_stride AssertionError) on BOTH the cached and inlined
        # paths. The eager backend is layout-flexible and accepts it.
        m = torch.nn.Linear(8, 5).eval()
        xex = torch.randn(
            8, 6
        ).t()  # example: shape (6, 8), non-contiguous stride (1, 6)
        self.assertFalse(xex.is_contiguous())
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, xex)]
        )
        self.assertIn("assert_size_stride", code)  # the layout guard we convert
        xrt = torch.randn(6, 8)  # same shape, contiguous -> different layout
        with self.assertRaisesRegex(PrecompileError, "memory format"):
            torch.compiler.precompile.load(code, cache)(m, xrt)  # cached path
        with self.assertRaisesRegex(PrecompileError, "memory format"):
            torch.compiler.precompile.load(code, _strip_artifact(cache))(
                m, xrt
            )  # inlined path
        # A matching (same-stride) input still works on inductor.
        xmatch = torch.randn(8, 6).t()
        self.assertEqual(
            torch.compiler.precompile.load(code, cache)(m, xmatch), m(xmatch)
        )
        # The eager backend accepts the differently-strided input.
        ecode, ecache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, xex)], backend="eager"
        )
        self.assertEqual(torch.compiler.precompile.load(ecode, ecache)(m, xrt), m(xrt))

    def test_input_layout_mismatch_enforced_without_size_asserts(self):
        # The layout guard must be a PROACTIVE driver check, not a reliance on inductor's
        # assert_size_stride: with size_asserts=False the assert is elided, so a naive
        # try/except would silently read wrong strides. Both load paths must still raise.
        import torch._inductor.config as ind_config

        m = torch.nn.Linear(8, 5).eval()
        xex = torch.randn(8, 6).t()  # non-contiguous example, shape (6, 8)
        xrt = torch.randn(6, 8)  # same shape, contiguous -> different layout
        with ind_config.patch(size_asserts=False):
            code, cache = torch.compiler.precompile(
                lambda model, t: model(t), example_inputs=[(m, xex)]
            )
            with self.assertRaisesRegex(PrecompileError, "memory format"):
                torch.compiler.precompile.load(code, cache)(m, xrt)  # cached path
            with self.assertRaisesRegex(PrecompileError, "memory format"):
                torch.compiler.precompile.load(code, _strip_artifact(cache))(
                    m, xrt
                )  # inlined

    def test_input_shape_mismatch_clean_error(self):
        # A same-structure but wrong-SHAPE input is an invariant-3 (shape) mismatch, NOT
        # an invariant-6 layout one: the driver must say "shape" / invariant 3 and not
        # misadvise a no-op .contiguous() (both inputs here are already contiguous).
        m = torch.nn.Linear(8, 5).eval()
        xex = torch.randn(6, 8)  # contiguous example
        xrt = torch.randn(7, 8)  # contiguous, different shape (same pytree structure)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, xex)]
        )
        with self.assertRaisesRegex(PrecompileError, "shape"):
            torch.compiler.precompile.load(code, cache)(m, xrt)  # cached path
        with self.assertRaisesRegex(PrecompileError, "shape"):
            torch.compiler.precompile.load(code, _strip_artifact(cache))(
                m, xrt
            )  # inlined path
        # The error must NOT mislabel a pure shape mismatch as a memory-format one.
        try:
            torch.compiler.precompile.load(code, cache)(m, xrt)
        except PrecompileError as e:
            self.assertNotIn("memory format", str(e))

    def test_size1_dim_stride_exempt_like_inductor(self):
        # A size-1 dim's stride is irrelevant (one element); inductor's assert_size_stride
        # ignores it (guards.cpp), so the proactive layout check must too -- a kept-dim
        # slice x[i:i+1] (size-1 dim with a wider stride) must RUN, not raise.
        m = torch.nn.Linear(4, 3).eval()
        xex = torch.randn(1, 4)  # contiguous, stride (4, 1)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, xex)]
        )
        row = torch.randn(2, 8)[
            0:1, :4
        ]  # shape (1, 4), stride (8, 1): size-1 dim differs
        self.assertEqual(tuple(row.shape), (1, 4))
        self.assertNotEqual(row.stride(), xex.stride())
        self.assertEqual(torch.compiler.precompile.load(code, cache)(m, row), m(row))
        self.assertEqual(
            torch.compiler.precompile.load(code, _strip_artifact(cache))(m, row),
            m(row),
        )

    def test_empty_input_shape_is_still_checked(self):
        # The numel==0 exemption must relax ONLY the (meaningless) stride check, not the
        # shape check: an empty runtime input whose shape differs from the example must
        # still raise invariant 3, not silently return the traced-shape output.
        code, cache = torch.compiler.precompile(
            lambda t: t.sum(0), example_inputs=[(torch.randn(0, 4),)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "shape"):
            f_c(torch.randn(0, 6))
        # A matching empty input runs (shape matches; stride is not checked).
        self.assertEqual(f_c(torch.randn(0, 4)), torch.randn(0, 4).sum(0))

    def test_shape_only_input_is_layout_flexible(self):
        # An input used only for its .shape (not its data) is not stride-consumed by the
        # kernel, so inductor emits no assert_size_stride for it; a transposed version
        # (same shape) must RUN, not be wrongly rejected as a memory-format mismatch.
        class M(torch.nn.Module):
            def forward(self, x, y):
                return y * x.shape[0]

        m = M().eval()
        x = torch.randn(4, 4)  # square so .t() keeps shape (4, 4)
        y = torch.randn(4, 4)
        code, cache = torch.compiler.precompile(
            lambda mm, a, b: mm(a, b), example_inputs=[(m, x, y)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        xt = x.t()  # same shape, different stride; only x.shape is consumed
        self.assertNotEqual(xt.stride(), x.stride())
        self.assertEqual(f_c(m, xt, y), m(xt, y))
        # A different x SHAPE is still rejected (x.shape[0] is baked).
        with self.assertRaisesRegex(PrecompileError, "shape"):
            f_c(m, torch.randn(5, 4), y)

    def test_dynamic_shapes_static_dim_still_checked(self):
        # The non-marked (feature) dim stays specialized: a mismatch on it is rejected,
        # while the marked (batch) dim is free.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0)
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, torch.randn(16, 4)).shape, (16, 3))  # dynamic dim free
        with self.assertRaisesRegex(PrecompileError, "dynamic dim"):
            f_c(m, torch.randn(16, 5))  # static feature dim mismatched

    def test_dynamic_shapes_guard_required_rejected(self):
        # A graph that must guard on the dynamic dim fails LOUDLY at capture (the unbacked
        # dim cannot be guarded), as a clear PrecompileError rather than a silent artifact.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0)

        def needs_guard(mm, t):
            if t.shape[0] > 4:
                return mm(t)
            return mm(t) + 1

        with self.assertRaisesRegex(PrecompileError, "guard on a dim marked with"):
            torch.compiler.precompile(needs_guard, example_inputs=[(m, x)])

    def test_dynamic_shapes_eager_rejected(self):
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0)
        with self.assertRaisesRegex(
            PrecompileError, "only supported with backend='inductor'"
        ):
            torch.compiler.precompile(
                lambda mm, t: mm(t), example_inputs=[(m, x)], backend="eager"
            )

    @parametrize("path", ("cached", "inlined"))
    def test_dtype_mismatch_rejected(self, path):
        # Each dense input's dtype is baked at capture (invariant 6); a runtime input of
        # a different dtype is rejected up front on BOTH the cached and inlined paths.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)  # float32 example
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)]
        )
        if path == "inlined":
            cache = _strip_artifact(cache)
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "dtype"):
            f_c(m, x.double())

    @unittest.skipUnless(TEST_CUDA, "needs CUDA for a cpu-vs-cuda device mismatch")
    @parametrize("path", ("cached", "inlined"))
    def test_device_mismatch_rejected(self, path):
        # Each dense input's device is baked at capture (invariant 6); a cpu-traced
        # artifact rejects a cuda input up front on BOTH load paths.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)  # cpu example
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)]
        )
        if path == "inlined":
            cache = _strip_artifact(cache)
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "device"):
            f_c(m, x.cuda())

    def test_mark_dynamic_backed_rejected(self):
        # Backed dynamic marks (mark_dynamic) have no analogue in the static/unbacked
        # capture path; precompile rejects them loudly rather than silently dropping
        # them and baking a wrong artifact (invariant 3).
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_dynamic(x, 0)
        with self.assertRaisesRegex(PrecompileError, "mark_dynamic"):
            torch.compiler.precompile(lambda mm, t: mm(t), example_inputs=[(m, x)])

    def test_mark_unbacked_hint_override_honored(self):
        # A mark_unbacked hint_override is a perf-only autotuning size hint (never a
        # guard), so precompile does NOT reject it; the single artifact is valid for any
        # runtime size and the hint is threaded onto the capture ShapeEnv's symbol.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0, hint_override=16)
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, x), m(x))
        x2 = torch.randn(32, 4)
        self.assertEqual(f_c(m, x2), m(x2))

    def test_mark_unbacked_specialize_on_rejected(self):
        # A mark_unbacked specialize_on list cannot be honored (precompile produces a
        # single artifact, not per-value specializations); it is rejected at capture.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0, specialize_on=[lambda t: t.shape[0] == 8])
        with self.assertRaisesRegex(PrecompileError, "specialize_on"):
            torch.compiler.precompile(lambda mm, t: mm(t), example_inputs=[(m, x)])

    def test_mark_unbacked_subclass_rejected(self):
        # A mark_unbacked dim on a tensor subclass (DTensor) cannot be honored: the
        # dynamic capture refakes a marked leaf via torch.empty, which drops the subclass
        # and would trace on a plain dense tensor. mark_unbacked stamps its marks on the
        # OUTER DTensor too (the decorator's DTensor branch falls through), so precompile
        # sees the mark and must reject it LOUDLY rather than silently tracing a
        # subclass-stripped tensor (invariant 3).
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("gloo not available")

        from torch.distributed.tensor import DeviceMesh, distribute_tensor, Replicate
        from torch.testing._internal.common_utils import find_free_port

        env = {"MASTER_ADDR": "localhost", "MASTER_PORT": str(find_free_port())}
        with mock.patch.dict(os.environ, env):
            dist.init_process_group("gloo", rank=0, world_size=1)
            try:
                mesh = DeviceMesh("cpu", list(range(1)))
                m = torch.nn.Linear(4, 3).eval()
                x = distribute_tensor(torch.randn(8, 4), mesh, [Replicate()])
                mark_unbacked(x, 0)
                with self.assertRaisesRegex(PrecompileError, "tensor subclass"):
                    torch.compiler.precompile(
                        lambda mm, t: mm(t), example_inputs=[(m, x)]
                    )
            finally:
                dist.destroy_process_group()

    @parametrize("path", ("cached", "inlined"))
    def test_shape_id_mismatched_sizes_rejected(self, path):
        # Two inputs sharing a shape_id reuse ONE unbacked symbol, so their marked dims
        # are equal by construction. A runtime call passing MISMATCHED sizes for those
        # dims violates the baked equality and is rejected with a clear PrecompileError.
        # The cached path catches it via the reconstructed artifact's assert_size_stride;
        # the inlined (artifact-stripped) path catches it via the inlined driver's own
        # assert_size_stride relabel -- exercise both so the inlined driver copy is covered.
        m = torch.nn.Linear(4, 4).eval()
        x = torch.randn(8, 4)
        y = torch.randn(8, 4)
        mark_unbacked(x, 0, shape_id="b")
        mark_unbacked(y, 0, shape_id="b")
        code, cache = torch.compiler.precompile(
            lambda mm, a, b: mm(a) + b, example_inputs=[(m, x, y)]
        )
        if path == "inlined":
            blob = torch.load(io.BytesIO(cache), weights_only=True)
            blob["artifact"] = None
            buf = io.BytesIO()
            torch.save(blob, buf)
            cache = buf.getvalue()
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "shape or memory format"):
            f_c(m, torch.randn(8, 4), torch.randn(16, 4))

    @parametrize("path", ("cached", "inlined"))
    def test_shape_id_bounds_from_both_occurrences_enforced(self, path):
        # Bounds from BOTH occurrences of a shared shape_id are applied to the single
        # shared symbol at capture: a min on one input and a max on the other are each
        # threaded onto the same unbacked symbol (see _fakeify_with_unbacked) AND baked as
        # a runtime USER_INPUT_BOUNDS guard. mark_unbacked's docstring promises a runtime
        # min/max check; this asserts it actually fires. An OUT-OF-BOUNDS size (< 2 or
        # > 64) is rejected with a PrecompileError naming the bound, while in-bounds sizes
        # (including the boundaries 2 and 64) still run and match eager. Both load paths.
        m = torch.nn.Linear(4, 4).eval()
        x = torch.randn(8, 4)
        y = torch.randn(8, 4)
        mark_unbacked(x, 0, shape_id="b", min=2)
        mark_unbacked(y, 0, shape_id="b", max=64)
        code, cache = torch.compiler.precompile(
            lambda mm, a, b: mm(a) + b, example_inputs=[(m, x, y)]
        )
        if path == "inlined":
            blob = torch.load(io.BytesIO(cache), weights_only=True)
            blob["artifact"] = None
            buf = io.BytesIO()
            torch.save(blob, buf)
            cache = buf.getvalue()
        f_c = torch.compiler.precompile.load(code, cache)
        for bs in (2, 8, 64):  # min boundary, an interior size, max boundary
            xt = torch.randn(bs, 4)
            yt = torch.randn(bs, 4)
            self.assertEqual(f_c(m, xt, yt), m(xt) + yt)
        # Below the declared min on the first occurrence's dim is rejected.
        with self.assertRaisesRegex(PrecompileError, "min=2"):
            f_c(m, torch.randn(1, 4), torch.randn(1, 4))
        # Above the declared max (from the second occurrence) is rejected.
        with self.assertRaisesRegex(PrecompileError, "max=64"):
            f_c(m, torch.randn(65, 4), torch.randn(65, 4))

    @parametrize("path", ("cached", "inlined"))
    def test_mark_unbacked_min_enforced_at_runtime(self, path):
        # mark_unbacked(x, 0, min=4) promises (in its docstring) a runtime check that the
        # dim is >= min. The capture-time torch._check on the unbacked symint never becomes
        # a runtime guard, so precompile bakes USER_INPUT_BOUNDS and the driver enforces it:
        # running the artifact at batch 2 raises a PrecompileError naming the bound on BOTH
        # load paths, while batch 8 runs and matches eager.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0, min=4)
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)]
        )
        self.assertIn("USER_INPUT_BOUNDS = [{0: (4, None)}]", code)
        if path == "inlined":
            blob = torch.load(io.BytesIO(cache), weights_only=True)
            blob["artifact"] = None
            buf = io.BytesIO()
            torch.save(blob, buf)
            cache = buf.getvalue()
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "size 2.*min=4"):
            f_c(m, torch.randn(2, 4))
        xt = torch.randn(8, 4)
        self.assertEqual(f_c(m, xt), m(xt))

    def test_eager_backend_wrong_static_shape_rejected(self):
        # The eager driver now checks USER_INPUT_SHAPES too: a wrong static shape is
        # rejected (invariant 3).
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)], backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "shape"):
            f_c(m, torch.randn(7, 4))

    def test_eager_backend_dtype_mismatch_rejected(self):
        # The eager driver checks USER_INPUT_DTYPES too: a dtype mismatch is rejected
        # (invariant 6).
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)], backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "dtype"):
            f_c(m, x.double())

    def test_cache_integrity_tampered_backend_rejected(self):
        # The cache envelope's backend tag is an integrity check: a tampered backend
        # (here flipped to a value that does not match python_code's BACKEND) makes
        # load() raise a clear PrecompileError rather than reconstruct a foreign cache.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)]
        )
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        blob["backend"] = "eager"  # python_code says inductor
        buf = io.BytesIO()
        torch.save(blob, buf)
        with self.assertRaisesRegex(PrecompileError, "backend"):
            torch.compiler.precompile.load(code, buf.getvalue())

    @parametrize("tag", ("format", "version"))
    def test_cache_format_version_mismatch_degrades(self, tag):
        # The cache is acceleration-only, so a FORMAT or VERSION mismatch (a foreign or
        # different-build envelope) is NOT fatal: load() DEGRADES to JIT'ing from
        # python_code rather than hard-failing. The reloaded callable must still run and
        # match eager, and load() must emit a degrade WARNING on the torch._precompile
        # logger. (A BACKEND or CODE_HASH mismatch still hard-fails -- see
        # test_cache_integrity_tampered_backend_rejected and
        # test_load_rejects_mismatched_code_cache_pair.)
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)]
        )
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        # Tamper either the format string or bump the version to a foreign value.
        blob[tag] = "not-a-precompile-cache" if tag == "format" else 999
        buf = io.BytesIO()
        torch.save(blob, buf)
        with self.assertLogs("torch._precompile", level="WARNING") as cm:
            f_c = torch.compiler.precompile.load(code, buf.getvalue())  # must not raise
        self.assertTrue(
            any("different torch build" in line for line in cm.output),
            f"expected a format/version degrade warning, got: {cm.output}",
        )
        self.assertEqual(f_c(m, x), m(x))  # JIT fallback runs and is correct

    def test_missing_calling_convention_metadata_rejected(self):
        # Syntactically valid python_code that lacks a required metadata global is not a
        # precompile artifact; load() raises a clear PrecompileError naming the gap.
        buf = io.BytesIO()
        torch.save(
            {
                "format": "torch.compiler.precompile",
                "version": 1,
                "backend": "inductor",
                "artifact": None,
            },
            buf,
        )
        with self.assertRaisesRegex(
            PrecompileError, "missing calling-convention metadata"
        ):
            torch.compiler.precompile.load("x = 1\n", buf.getvalue())

    def test_load_non_literal_metadata_rejected(self):
        # A truncated / hand-edited artifact can leave a metadata name assigned
        # a non-literal expression; _parse_artifact_metadata must surface the
        # documented PrecompileError naming the metadata (not ast's raw
        # "malformed node" ValueError).
        with self.assertRaisesRegex(
            PrecompileError, "non-literal metadata to 'BACKEND'"
        ):
            torch.compiler.precompile.load(
                "BACKEND = str(1)\nTRACER = 'make_fx'\n", b""
            )

    def test_singleton_pickle_deepcopy_roundtrip(self):
        # torch.compiler.precompile is a process-wide singleton; pickle and deepcopy
        # must round-trip to the SAME object (it carries no per-call state), and its
        # repr is the stable public name.
        p = torch.compiler.precompile
        self.assertIs(pickle.loads(pickle.dumps(p)), p)
        self.assertIs(copy.deepcopy(p), p)
        self.assertEqual(repr(p), "torch.compiler.precompile")

    def test_standalone_runtime_artifact_execs_in_fresh_process(self):
        # A generated artifact that imports a standalone_runtime helper (here output-
        # aliasing, which emits ``from ...standalone_runtime import gen_alias_from_base``)
        # must EXEC in a FRESH process whose only prior import is ``torch`` -- a
        # regression for the runtime_wrappers <-> _dynamo circular import that a cold
        # exec used to hit. We write python_code to a temp file and exec it in a
        # subprocess that imports only torch, then runs forward().
        x = torch.randn(3, 4)
        code, _cache = torch.compiler.precompile(lambda a: a.t(), example_inputs=[(x,)])
        self.assertIn("standalone_runtime import gen_alias_from_base", code)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False
        ) as artifact_file:
            artifact_file.write(code)
            artifact_path = artifact_file.name
        driver = textwrap.dedent(
            f"""
            import torch  # the ONLY pre-import; the artifact must self-bootstrap
            ns = {{"__name__": "_fresh_artifact"}}
            with open({artifact_path!r}) as fh:
                exec(compile(fh.read(), {artifact_path!r}, "exec"), ns)
            x = torch.randn(3, 4)
            out = ns["forward"](x)
            assert torch.equal(out, x.t()), "fresh-process artifact output mismatch"
            print("FRESH_OK")
            """
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", driver],
                capture_output=True,
                text=True,
                timeout=300,
            )
        finally:
            if os.path.exists(artifact_path):
                os.remove(artifact_path)
        self.assertEqual(
            proc.returncode, 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
        self.assertIn("FRESH_OK", proc.stdout)

    def test_load_rejects_mismatched_code_cache_pair(self):
        # The cache envelope's code_hash (sha256 of python_code) binds a cache to the
        # EXACT python_code it accelerates. Two artifacts from the SAME backend but
        # DIFFERENT fn produce different python_code (hence different code_hash), so
        # pairing one's code with the other's cache must fail loudly rather than
        # silently run the cache's compiled graph under foreign metadata (the core
        # silent-wrong-result guard). The MATCHED pair still runs and is correct.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        codeA, cacheA = torch.compiler.precompile(
            lambda mm, t: mm(t) * 2, example_inputs=[(m, x)]
        )
        codeB, cacheB = torch.compiler.precompile(
            lambda mm, t: mm(t) + 100, example_inputs=[(m, x)]
        )
        self.assertNotEqual(codeA, codeB)
        with self.assertRaisesRegex(PrecompileError, "code_hash|does not match"):
            torch.compiler.precompile.load(codeA, cacheB)
        f_a = torch.compiler.precompile.load(codeA, cacheA)
        self.assertEqual(f_a(m, x), m(x) * 2)

    def test_non_size_stride_assertion_propagates_unchanged(self):
        # The inductor driver's forward() wraps the inlined ``call`` in a try/except
        # AssertionError that relabels ONLY inductor's own assert_size_stride failure
        # (a layout/shape mismatch) as a "shape or memory format" PrecompileError. A
        # NON-size-stride AssertionError (e.g. a user torch._assert or an internal
        # invariant) must propagate with its ORIGINAL message, not be mislabeled. A
        # call() that raises a non-layout AssertionError is hard to trigger from a real
        # compiled artifact, so doctor a real artifact's call() to raise a custom
        # assertion and re-pair its code_hash, exercising the inlined relabel guard.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)]
        )
        head = code[: code.index("\ndef call(")]
        banner = code.rindex(
            "# " + "=" * 70, 0, code.index("# 2. Calling-convention metadata")
        )
        new_call = (
            '\n\ndef call(args):\n    assert False, "my custom user assertion"\n\n\n'
        )
        new_code = head + new_call + code[banner:]
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        blob["artifact"] = None  # force the inlined path so the doctored call() runs
        import hashlib

        blob["code_hash"] = hashlib.sha256(new_code.encode()).hexdigest()
        buf = io.BytesIO()
        torch.save(blob, buf)
        f = torch.compiler.precompile.load(new_code, buf.getvalue())
        with self.assertRaisesRegex(AssertionError, "my custom user assertion"):
            f(m, x)
        # The original assertion must NOT be relabeled as a layout error.
        try:
            f(m, x)
        except AssertionError as e:
            self.assertNotIn("shape or memory format", str(e))

    def test_public_identity_module_and_qualname(self):
        # PrecompileError and load are public under torch.compiler.precompile, so their
        # __module__ / __qualname__ must report that public location (so Sphinx and
        # introspection anchor them under torch.compiler, not the private module).
        err = torch.compiler.precompile.PrecompileError
        self.assertEqual(err.__module__, "torch.compiler")
        self.assertEqual(err.__qualname__, "precompile.PrecompileError")
        self.assertEqual(torch.compiler.precompile.load.__module__, "torch.compiler")
        self.assertEqual(torch.compiler.precompile.load.__qualname__, "precompile.load")
        for name in (
            "load_files",
            "stateful",
            "PrecompileState",
            "PrecompileStateSummary",
        ):
            member = getattr(torch.compiler.precompile, name)
            self.assertEqual(member.__module__, "torch.compiler")
            self.assertEqual(member.__qualname__, f"precompile.{name}")

    @parametrize("backend", ("inductor", "eager"))
    def test_renamed_buffer_structural_mismatch_rejected(self, backend):
        # The BUFFER_NAMES half of the structural check (invariant 2): a runtime model
        # whose PARAM names match exactly but a BUFFER is renamed (same count and shape)
        # must be rejected, since the buffer name list is part of the baked structure.
        # The cached/inlined inductor driver and the eager driver each have their own
        # _check_structure, so cover both backends.
        class WithBuf(torch.nn.Module):
            def __init__(self, bufname):
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)
                self.register_buffer(bufname, torch.randn(3))
                self._bn = bufname

            def forward(self, x):
                return self.lin(x) + getattr(self, self._bn)

        m = WithBuf("buf").eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)], backend=backend
        )
        self.assertIn("BUFFER_NAMES = ['buf']", code)
        renamed = WithBuf("buf2").eval()  # same params, buffer renamed (same shape)
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "do not match the traced model"):
            f_c(renamed, x)

    def test_example_input_inplace_mutation_not_restored(self):
        # Capture EXECUTES fn once on the example inputs (invariant 3), so an in-place
        # mutation fn performs on its example user input happens at capture time and is
        # NOT restored -- only .grad is snapshotted/restored. Pin this surprising contract
        # so it stays covered: the example tensor reflects the mutation afterward.
        scratch = torch.zeros(4)
        torch.compiler.precompile(lambda a: a.add_(1.0), example_inputs=[(scratch,)])
        self.assertEqual(scratch, torch.ones(4))

    @parametrize("path", ("cached", "inlined", "eager"))
    def test_wrong_dtype_rejected_across_all_paths(self, path):
        # The same wrong-dtype input is rejected on ALL load paths -- cached (artifact),
        # inlined (artifact stripped), and eager -- each with its own driver copy of the
        # dtype check (invariant 6). Loading the SAME inductor artifact via cached and
        # inlined, plus a separate eager artifact, keeps the three drivers in agreement.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        if path == "eager":
            code, cache = torch.compiler.precompile(
                lambda mm, t: mm(t), example_inputs=[(m, x)], backend="eager"
            )
        else:
            code, cache = torch.compiler.precompile(
                lambda mm, t: mm(t), example_inputs=[(m, x)]
            )
            if path == "inlined":
                cache = _strip_artifact(cache)
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "dtype"):
            f_c(m, x.double())

    @unittest.skipUnless(TEST_CUDA, "needs CUDA for a cpu-vs-cuda device mismatch")
    def test_eager_device_mismatch_rejected(self):
        # The eager driver bakes each input's device (invariant 6): a cpu-traced eager
        # artifact rejects a cuda input up front, like the inductor backend.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)  # cpu example
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)], backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "device"):
            f_c(m, x.cuda())

    def test_unserializable_in_spec_accepts_distinct_structures(self):
        # When IN_SPEC degrades to None (the input pytree spec was not serializable) the
        # structural in_spec check is SKIPPED -- a documented best-effort limit. Two
        # SAME-leaf-count, same-per-leaf-shape but STRUCTURALLY DISTINCT runtime inputs
        # are therefore both accepted without error (the only check left is leaf count /
        # per-leaf shape). Make that best-effort gap explicit.
        m = torch.nn.Linear(4, 3).eval()
        inp = _UnserializableCtxInput(torch.randn(5, 4), torch.randn(5, 4))
        code, cache = torch.compiler.precompile(
            lambda model, h: model(h.a + h.b), example_inputs=[(m, inp)]
        )
        self.assertIn("IN_SPEC = None", code)
        f_c = torch.compiler.precompile.load(code, cache)
        t = torch.randn(5, 4)
        # The traced structure (the custom node) and a plain list of the same two leaves
        # have distinct pytree structures but the same flattened leaves/shapes; both run.
        out_node = f_c(m, _UnserializableCtxInput(t, t))
        out_list = f_c(m, [t, t])
        self.assertEqual(out_node, m(t + t))
        self.assertEqual(out_list, m(t + t))

    @parametrize("path", ("cached", "inlined"))
    def test_mark_unbacked_max_enforced_at_runtime(self, path):
        # The max-only mirror of test_mark_unbacked_min_enforced_at_runtime:
        # mark_unbacked(x, 0, max=16) records USER_INPUT_BOUNDS = [{0: (None, 16)}] and
        # the driver rejects an ABOVE-max runtime size on BOTH load paths (the capture-time
        # torch._check never becomes a runtime guard on an unbacked symint), while an
        # in-bounds size runs and matches eager.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0, max=16)
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)]
        )
        self.assertIn("USER_INPUT_BOUNDS = [{0: (None, 16)}]", code)
        if path == "inlined":
            blob = torch.load(io.BytesIO(cache), weights_only=True)
            blob["artifact"] = None
            buf = io.BytesIO()
            torch.save(blob, buf)
            cache = buf.getvalue()
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "max"):
            f_c(m, torch.randn(32, 4))
        xt = torch.randn(8, 4)
        self.assertEqual(f_c(m, xt), m(xt))

    @unittest.skipUnless(TEST_CUDA, "functionalize_rng_ops seeds via CUDA rng state")
    def test_functionalized_rng_matches_eager_cpu(self):
        # Under functionalized RNG the dropout draw is seeded from the global generator,
        # so seeding torch.manual_seed identically before the artifact run and before eager
        # makes both draw the SAME dropout mask: the artifact output is numerically EQUAL
        # to eager (a stronger check than structure-only). This runs on CPU tensors, but
        # functionalize_rng_ops still seeds via CUDARngStateHelper.get_torch_state_as_tuple,
        # which raises unless CUDA is available, so the whole test is gated on TEST_CUDA
        # (mirroring test_functionalized_rng_supported). The CUDA functionalized path uses
        # different Philox offset bookkeeping than eager, so this numeric equivalence is
        # CPU-tensor-only (see test_functionalized_rng_supported for the device-generic
        # structural check).
        import torch._functorch.config as functorch_config

        x = torch.randn(64)
        with functorch_config.patch(functionalize_rng_ops=True):
            code, cache = torch.compiler.precompile(
                lambda a: torch.nn.functional.dropout(a, 0.5, training=True),
                example_inputs=[(x,)],
            )
            f_c = torch.compiler.precompile.load(code, cache)
            torch.manual_seed(0)
            out = f_c(x)
        torch.manual_seed(0)
        ref = torch.nn.functional.dropout(x, 0.5, training=True)
        self.assertTrue((out == 0).any())  # dropout zeroed some elements
        self.assertEqual(out, ref)  # same mask under the same seed

    @parametrize("backend", ("inductor", "eager"))
    def test_param_shape_mismatch_rejected(self, backend):
        # The headline silent-wrong-result fix: the structural check (invariant 2) now
        # compares each runtime param's SHAPE against the baked example, not just its
        # name/count. A runtime model with the SAME param names but a different param
        # SHAPE (here Linear(4, K) for the traced Linear(4, M), K != M) is rejected with a
        # PrecompileError naming the offending param -- on BOTH backends, and on the
        # inductor backend's cached AND inlined load paths. Before the fix the eager
        # backend (no assert_size_stride backstop) silently returned a wrong-shaped tensor.
        m = torch.nn.Linear(4, 3).eval()  # M = 3
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)], backend=backend
        )
        bad = torch.nn.Linear(4, 7).eval()  # K = 7 != 3, same param names

        for label, f_c in _default_and_inlined_loaders(code, cache, backend):
            with self.subTest(path=label):
                with self.assertRaisesRegex(PrecompileError, "weight.*shape"):
                    f_c(bad, x)

    @parametrize("backend", ("inductor", "eager"))
    def test_param_dtype_mismatch_rejected(self, backend):
        # The dtype half of the structural shape/dtype check (invariant 2): a runtime
        # model with the SAME param names and shapes but a different param DTYPE (a
        # .half() copy of the traced float32 model) is rejected with a PrecompileError
        # naming the param, on both backends, AND -- on the inductor backend -- on the
        # cached (artifact) AND inlined (artifact-stripped) load paths. The inlined
        # inductor driver has its own _check_structure dtype branch, so cover it the
        # same way test_param_shape_mismatch_rejected covers the shape branch.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)], backend=backend
        )
        bad = torch.nn.Linear(4, 3).eval().half()  # same shape, different dtype

        for label, f_c in _default_and_inlined_loaders(code, cache, backend):
            with self.subTest(path=label):
                with self.assertRaisesRegex(PrecompileError, "weight.*dtype"):
                    f_c(bad, x)

    @parametrize("backend", ("inductor", "eager"))
    def test_buffer_shape_dtype_mismatch_rejected(self, backend):
        # The BUFFER half of the structural SHAPE/DTYPE check (invariant 2): the
        # structural loop iterates PARAM_NAMES then BUFFER_NAMES, but only the param
        # branch was exercised elsewhere. A runtime model whose PARAMS match exactly but
        # whose registered BUFFER (same name, same count) has a different SHAPE or DTYPE
        # must be rejected naming that buffer. Cover both backends, and -- on inductor --
        # the cached AND inlined driver copies (each has its own _check_structure).
        class WithBuf(torch.nn.Module):
            def __init__(self, size, dtype):
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)
                # A plain buffer the graph READS, so it is lifted to a graph input and
                # survives to the structural check (a buffer never read might be elided).
                self.register_buffer("b", torch.randn(size).to(dtype))

            def forward(self, x):
                return self.lin(x) + self.b.sum()

        m = WithBuf(3, torch.float32).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)], backend=backend
        )
        self.assertIn("BUFFER_NAMES = ['b']", code)
        # Same buffer name and count, but a different SHAPE / DTYPE.
        bad_shape = WithBuf(5, torch.float32).eval()
        bad_dtype = WithBuf(3, torch.float64).eval()

        for label, f_c in _default_and_inlined_loaders(code, cache, backend):
            with self.subTest(path=label):
                with self.assertRaisesRegex(PrecompileError, r"'b'.*shape"):
                    f_c(bad_shape, x)
                with self.assertRaisesRegex(PrecompileError, r"'b'.*dtype"):
                    f_c(bad_dtype, x)

    def test_param_layout_specialization_rejected_inductor(self):
        # MAJOR2 (invariant 2 inductor caveat / invariant 6): the inductor backend bakes
        # each param/buffer's LAYOUT (memory format) too, since it emits assert_size_stride
        # on every weight the graph reads. A runtime model whose weight has the SAME
        # shape+dtype but a DIFFERENT memory format (a non-contiguous view) is rejected,
        # with the broadened relabel that names a model PARAMETER/BUFFER layout. The eager
        # backend is layout-flexible and ACCEPTS the same non-contiguous weight.
        m = torch.nn.Linear(8, 5).eval()
        x = torch.randn(4, 8)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)]
        )

        def with_noncontig_weight():
            run = torch.nn.Linear(8, 5).eval()
            run.load_state_dict(m.state_dict())
            # A non-contiguous view of the same data: same shape+dtype, different layout.
            nc = run.weight.data.t().contiguous().t()
            self.assertFalse(nc.is_contiguous())
            self.assertEqual(tuple(nc.shape), tuple(m.weight.shape))
            run.weight = torch.nn.Parameter(nc)
            return run

        def loaders():
            yield "cached", torch.compiler.precompile.load(code, cache)
            yield (
                "inlined",
                torch.compiler.precompile.load(code, _strip_artifact(cache)),
            )

        for label, f_c in loaders():
            with self.subTest(path=label):
                with self.assertRaisesRegex(
                    PrecompileError, r"memory format.*PARAMETER/BUFFER.*layout"
                ):
                    f_c(with_noncontig_weight(), x)
        # The eager backend accepts the same non-contiguous weight (layout-flexible).
        ecode, ecache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)], backend="eager"
        )
        run = with_noncontig_weight()
        self.assertEqual(torch.compiler.precompile.load(ecode, ecache)(run, x), run(x))

    def test_unbacked_equality_shared_vs_independent_shape_id(self):
        # MAJOR1 (invariant 3 DANGER note): two mark_unbacked dims that the graph requires
        # to be EQUAL behave differently depending on shape_id. (a) A SHARED shape_id binds
        # them to ONE symbol, so they are equal by construction AND a runtime size mismatch
        # is LOUDLY rejected. (b) Two INDEPENDENTLY marked dims (no shared shape_id)
        # combined elementwise bake a SILENT equal-size assumption: unlike eager, a runtime
        # mismatch is NOT loudly rejected -- NOT because the constraint is unrecoverable, but
        # because precompile does not harvest it: the capture ShapeEnv DOES record the
        # equality as a deferred runtime assert (Eq(u0, u1)), yet only the decorator's
        # min/max feed USER_INPUT_BOUNDS, so the driver never enforces the relational assert.
        # The artifact runs and returns the FIRST input's shape. This documents the "give
        # equal-must-be-equal dims a shared shape_id" limitation (and would flip to a loud
        # failure if that harvesting gap is later closed) rather than asserting silent-wrong
        # is correct.
        m = torch.nn.Linear(4, 4).eval()
        # (a) shared shape_id -> equality enforced.
        xs = torch.randn(8, 4)
        ys = torch.randn(8, 4)
        mark_unbacked(xs, 0, shape_id="b")
        mark_unbacked(ys, 0, shape_id="b")
        code_s, cache_s = torch.compiler.precompile(
            lambda mm, a, b: mm(a) + b, example_inputs=[(m, xs, ys)]
        )
        f_s = torch.compiler.precompile.load(code_s, cache_s)
        xt, yt = torch.randn(8, 4), torch.randn(8, 4)
        self.assertEqual(f_s(m, xt, yt), m(xt) + yt)  # matched sizes work
        with self.assertRaisesRegex(PrecompileError, "shape or memory format"):
            f_s(m, torch.randn(8, 4), torch.randn(16, 4))  # mismatch rejected
        # (b) independent marks -> the documented silent equal-size limitation. A matched
        # call works; a mismatched call does NOT raise and returns the first input's shape.
        xi = torch.randn(8, 4)
        yi = torch.randn(8, 4)
        mark_unbacked(xi, 0)
        mark_unbacked(yi, 0)
        code_i, cache_i = torch.compiler.precompile(
            lambda mm, a, b: mm(a) + b, example_inputs=[(m, xi, yi)]
        )
        f_i = torch.compiler.precompile.load(code_i, cache_i)
        xm, ym = torch.randn(10, 4), torch.randn(10, 4)
        self.assertEqual(f_i(m, xm, ym), m(xm) + ym)  # matched sizes work
        out = f_i(m, torch.randn(10, 4), torch.randn(12, 4))  # mismatch NOT rejected
        self.assertEqual(tuple(out.shape), (10, 4))  # broadcasts to the first input

    def test_grad_identity_preserved_across_precompile(self):
        # Capture snapshots and restores the example model's .grad by the SAME object (no
        # clone), so a caller holding a prior p.grad reference -- or optimizer state keyed
        # on grad identity -- is not invalidated. Warm up a backward to populate .grad,
        # snapshot the object identity, precompile a backward step on the same model, and
        # assert p.grad is still the SAME object afterward.
        torch.manual_seed(0)
        m = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        m(x).sum().backward()  # warmup populates .grad
        g = m.weight.grad
        self.assertIsNotNone(g)
        torch.compiler.precompile(
            lambda mm, t: mm(t).sum().backward(), example_inputs=[(m, x)]
        )
        self.assertIs(m.weight.grad, g)  # same object, not a clone

    def test_precompile_error_public_binding(self):
        # PrecompileError is a single public type reachable two ways
        # (torch.compiler.PrecompileError and torch.compiler.precompile.PrecompileError),
        # is a real exception type, is advertised in torch.compiler.__all__, and a raised
        # instance is catchable via the public torch.compiler.PrecompileError alias.
        self.assertIs(
            torch.compiler.PrecompileError, torch.compiler.precompile.PrecompileError
        )
        self.assertIsInstance(torch.compiler.PrecompileError, type)
        self.assertIn("PrecompileError", torch.compiler.__all__)
        # A real PrecompileError (here the invariant-1 constant-tensor guard) is catchable
        # via the public torch.compiler.PrecompileError alias.
        captured = torch.randn(3)
        with self.assertRaisesRegex(torch.compiler.PrecompileError, "hard-coded"):
            torch.compiler.precompile(
                lambda x: x + captured, example_inputs=[(torch.randn(3),)]
            )

    def test_single_trust_warning_on_inlined_load(self):
        # On the inlined load path (an eager artifact has an empty cache, so there is
        # nothing to prime and load() just EXECs python_code) the untrusted-input / EXEC
        # warning must fire EXACTLY ONCE -- load() warns once, before cache processing.
        # Asserting "exactly once" guards against the warning being duplicated.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)], backend="eager"
        )
        with self.assertLogs("torch._precompile", level="WARNING") as cm:
            torch.compiler.precompile.load(code, cache)
        exec_warnings = [line for line in cm.output if "EXEC" in line]
        self.assertEqual(
            len(exec_warnings), 1, f"expected one EXEC warning, got: {cm.output}"
        )
        self.assertTrue(any("untrusted" in line.lower() for line in cm.output))

    def test_tied_weights_single_input_single_grad(self):
        # Invariants 1/2/5: a weight tied across two layers is interned by identity to a
        # SINGLE graph input (PARAM_NAMES lists the first name once) and accumulates ONE
        # grad -- the sum of both uses -- matching an eager backward, not one grad per name.
        class Tied(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = torch.nn.Linear(4, 4, bias=False)
                self.l2 = torch.nn.Linear(4, 4, bias=False)
                self.l2.weight = self.l1.weight  # tie: same tensor, two names

            def forward(self, x):
                return self.l2(self.l1(x))

        m = Tied()
        t = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t).sum().backward(), example_inputs=[(m, t)]
        )
        self.assertIn("PARAM_NAMES = ['l1.weight']", code)  # tie collapsed to one

        ref = copy.deepcopy(m)  # deepcopy preserves the tie within the object graph
        ref(t).sum().backward()

        torch.compiler.precompile.load(code, cache)(m, t)  # one call: tied grad
        self.assertEqual(m.l1.weight.grad, ref.l1.weight.grad)
        self.assertIs(m.l1.weight, m.l2.weight)  # still one tensor at runtime

    def test_multiple_module_args_all_lifted(self):
        # The multi=True naming branch: two DIFFERENT nn.Module args are BOTH lifted, their
        # positions recorded in MODULE_POSITIONS, and their params disambiguated as m0.* /
        # m1.* (per-module prefixes). Loaded artifact matches eager m2(m1(t)).
        torch.manual_seed(0)
        m1 = torch.nn.Linear(4, 4)
        m2 = torch.nn.Linear(4, 3)
        t = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda a, b, t: b(a(t)), example_inputs=[(m1, m2, t)]
        )
        self.assertIn("MODULE_POSITIONS = [0, 1]", code)
        self.assertIn("m0.weight", code)  # first module's params prefixed m0.*
        self.assertIn("m1.weight", code)  # second module's params prefixed m1.*
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m1, m2, t), m2(m1(t)))

    def test_frozen_param_keeps_none_grad(self):
        # Invariant 5 with a mix: only params that received a gradient are harvested
        # (recorded in GRAD_PARAM_INDICES), so a frozen (requires_grad=False) param keeps
        # .grad is None while a trainable param gets a grad matching an eager backward.
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.frozen = torch.nn.Linear(4, 4)
                self.trainable = torch.nn.Linear(4, 4)
                for p in self.frozen.parameters():
                    p.requires_grad_(False)

            def forward(self, x):
                return self.trainable(self.frozen(x))

        m = M()
        t = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t).sum().backward(), example_inputs=[(m, t)]
        )

        ref = copy.deepcopy(m)
        ref(t).sum().backward()

        torch.compiler.precompile.load(code, cache)(m, t)
        for p in m.frozen.parameters():
            self.assertIsNone(p.grad)  # frozen: never harvested
        for p in m.trainable.parameters():
            self.assertIsNotNone(p.grad)
        for (n, p), (_, rp) in zip(
            m.trainable.named_parameters(), ref.trainable.named_parameters()
        ):
            self.assertEqual(p.grad, rp.grad, n)

    def test_requires_grad_flip_is_noop(self):
        # Which params get a scattered grad is fixed at CAPTURE time from the example
        # model's requires_grad (invariant 5); flipping a runtime param's requires_grad
        # does NOT change what the artifact computes. Capture with params requiring grad,
        # set requires_grad=False on the runtime model, and assert the grad is STILL
        # scattered (and matches eager) -- locking the documented contract.
        torch.manual_seed(0)
        m = torch.nn.Linear(4, 3)  # params require grad at capture
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t).sum().backward(), example_inputs=[(m, x)]
        )
        run = torch.nn.Linear(4, 3)
        run.load_state_dict(m.state_dict())
        for p in run.parameters():
            p.requires_grad_(False)  # flip OFF at runtime -- must be a no-op
        torch.compiler.precompile.load(code, cache)(run, x)
        self.assertIsNotNone(run.weight.grad)  # still scattered despite the flip
        ref = torch.nn.Linear(4, 3)
        ref.load_state_dict(m.state_dict())
        ref(x).sum().backward()
        self.assertEqual(run.weight.grad, ref.weight.grad)

    def test_tracer_dynamo_rejects_tensor_reachable_through_helper_function(self):
        # A tensor Dynamo reads through an inlined helper -- from the helper's
        # defaults, or from a global only the helper's own code loads -- is a
        # graph placeholder rooted in the environment, so the exact placeholder
        # check rejects it (it would otherwise serve a raw NameError).
        for fn, name in (
            (_precompile_dynamo_calls_helper, "helper_with_default"),
            (_precompile_dynamo_calls_inlined_helper, "inlined_helper"),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    PrecompileError, "from the Python environment"
                ):
                    torch.compiler.precompile(
                        fn,
                        example_inputs=[(torch.randn(4),)],
                        tracer="dynamo",
                        backend="eager",
                    )

    def test_tracer_dynamo_rejects_storage_overlap_inside_object_input(self):
        # The overlap scan enumerates deeply: overlapping (and same-storage)
        # views inside a custom object must be rejected like bare view args.
        buf = torch.randn(8)
        for label, pair in (
            ("overlapping", _PrecompileDynamoTensorPair(buf[:4], buf[3:7])),
            ("same_storage", _PrecompileDynamoTensorPair(buf[:4], buf[4:])),
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    PrecompileError, "share or overlap storage"
                ):
                    torch.compiler.precompile(
                        _precompile_dynamo_pair_sum,
                        example_inputs=[(pair,)],
                        tracer="dynamo",
                        backend="eager",
                    )

    def test_tracer_dynamo_loaded_artifact_rejects_overlap_inside_object(self):
        # The emitted driver's runtime overlap check is the same walk as the
        # capture-side one (one implementation, emitted verbatim), so a tensor
        # inside a custom argument is seen; it runs because the graph mutates.
        good = _PrecompileDynamoTensorPair(torch.randn(4), torch.randn(4))
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_pair_inplace,
            example_inputs=[(good,)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        p = _PrecompileDynamoTensorPair(torch.randn(4), torch.randn(4))
        ref = _PrecompileDynamoTensorPair(p.a.clone(), p.b.clone())
        self.assertEqual(loaded(p), _precompile_dynamo_pair_inplace(ref))
        buf = torch.randn(8)
        bad = _PrecompileDynamoTensorPair(buf[:4], buf[4:])
        with self.assertRaisesRegex(PrecompileError, "share or overlap storage"):
            loaded(bad)

    def test_tracer_dynamo_walk_descends_dict_and_slots(self):
        # _instance_values is the SINGLE shared walk (torch._precompile_driver,
        # emitted verbatim into the artifact). Pin that it descends both a custom
        # object's __dict__ and its __slots__, and that BOTH consumers see nested
        # tensors/modules either way: the capture-side nn.Module rejection, the
        # capture-side storage-overlap check, and the driver's runtime overlap
        # check. (Module-in-__slots__ is covered by
        # test_tracer_dynamo_rejects_module_carried_in_argument_slot; overlap in
        # __dict__ by the two tests above.)
        buf = torch.randn(8)
        # Capture-side: overlapping tensors reached through __slots__ are rejected.
        with self.assertRaisesRegex(PrecompileError, "share or overlap storage"):
            torch.compiler.precompile(
                _precompile_dynamo_pair_sum,
                example_inputs=[(_PrecompileDynamoSlottedPair(buf[:4], buf[3:7]),)],
                tracer="dynamo",
                backend="eager",
            )
        # Capture-side: an nn.Module reached through a plain __dict__ is rejected.
        with self.assertRaisesRegex(PrecompileError, "nn.Module arguments"):
            torch.compiler.precompile(
                _precompile_dynamo_slotted_box_call,
                example_inputs=[(torch.randn(2), _PrecompileDynamoModuleInDict())],
                tracer="dynamo",
                backend="eager",
            )
        # Driver-side: the same emitted walk sees tensors in __slots__ too. Capture
        # a mutating fn on non-overlapping slotted tensors, then a runtime slotted
        # pair whose tensors overlap must be rejected at serve time.
        good = _PrecompileDynamoSlottedPair(torch.randn(4), torch.randn(4))
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_pair_inplace,
            example_inputs=[(good,)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "share or overlap storage"):
            loaded(_PrecompileDynamoSlottedPair(buf[:4], buf[4:]))

    def test_tracer_dynamo_mutating_container_input_captures_and_serves(self):
        # Guard minimization re-checks recorded examples after they ran; a
        # pre-execution snapshot keeps a fn that mutates a container input
        # capturable (plain torch.compile supports it).
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_mutating_step,
            example_inputs=[([1, 2], torch.randn(2))],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        t = torch.randn(2)
        xs = [1, 2]
        self.assertEqual(loaded(xs, t), t * 3)
        self.assertEqual(xs, [1, 2, 1])

    def test_tracer_dynamo_stateful_mutating_step_keeps_capturing(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _stateful_paths(tmp)
            t = torch.randn(2)
            results, state = torch.compiler.precompile.stateful(
                _precompile_dynamo_mutating_step,
                example_inputs=[([1, 2], t)],
                state=None,
                backend="eager",
                **paths,
            )
            self.addCleanup(state.close)
            self.assertIsInstance(results, list)
            self.assertEqual(results, [t * 3])
            # A later call still records and rewrites: the recorded snapshot,
            # not the mutated live list, is re-checked on rebuild.
            [result], state = torch.compiler.precompile.stateful(
                _precompile_dynamo_mutating_step,
                example_inputs=[([5, 6, 7], t)],
                state=state,
                backend="eager",
                **paths,
            )
            self.assertEqual(result, t * 4)
            self.assertEqual(state.summary().calls, 2)
            loaded = torch.compiler.precompile.load_files(**paths)
            self.assertEqual(loaded([1, 2], t), t * 3)

    def test_tracer_dynamo_stateful_survives_input_metadata_mutation(self):
        # The recorded example snapshot freezes tensor METADATA (a storage
        # alias owning its sizes/strides), so a caller that resize_()s an
        # input between calls cannot poison the state: rebuilds re-check every
        # recorded example, and a live-recorded tensor would re-check at the
        # mutated shape and fail every later call.
        with tempfile.TemporaryDirectory() as tmp:
            paths = _stateful_paths(tmp)
            buf = torch.randn(4)
            [r1], state = torch.compiler.precompile.stateful(
                _precompile_dynamo_torch_sin,
                example_inputs=[(buf,)],
                state=None,
                backend="eager",
                **paths,
            )
            self.addCleanup(state.close)
            self.assertEqual(r1, torch.sin(buf))
            buf.resize_(8)  # caller-side metadata mutation of call 1's example
            buf.fill_(0.5)
            [r2], state = torch.compiler.precompile.stateful(
                _precompile_dynamo_torch_sin,
                example_inputs=[(buf,)],
                state=state,
                backend="eager",
                **paths,
            )
            self.assertEqual(r2, torch.sin(buf))
            x3 = torch.randn(4)
            [r3], state = torch.compiler.precompile.stateful(
                _precompile_dynamo_torch_sin,
                example_inputs=[(x3,)],
                state=state,
                backend="eager",
                **paths,
            )
            self.assertEqual(r3, torch.sin(x3))
            summary = state.summary()
            self.assertEqual((summary.calls, summary.examples), (3, 3))
            loaded = torch.compiler.precompile.load_files(**paths)
            y = torch.randn(4)
            self.assertEqual(loaded(y), torch.sin(y))

    def test_tracer_dynamo_stateful_survives_transpose_between_calls(self):
        # The in-place-mutating-fn flavor of the metadata-freeze regression:
        # the step mutates its input DATA (add_) and the caller transpose_()s
        # the same tensor between calls (a metadata mutation). Both calls must
        # capture and rewrite; the frozen snapshot keeps call 1's example at
        # its original metadata.
        with tempfile.TemporaryDirectory() as tmp:
            paths = _stateful_paths(tmp)
            x = torch.randn(2, 3)
            expected = (x.clone() + 1) * 2
            [r1], state = torch.compiler.precompile.stateful(
                _precompile_dynamo_inplace_step,
                example_inputs=[(x,)],
                state=None,
                backend="eager",
                **paths,
            )
            self.addCleanup(state.close)
            self.assertEqual(r1, expected)
            x.transpose_(0, 1)  # caller-side metadata mutation between calls
            expected = (x.clone() + 1) * 2
            [r2], state = torch.compiler.precompile.stateful(
                _precompile_dynamo_inplace_step,
                example_inputs=[(x,)],
                state=state,
                backend="eager",
                **paths,
            )
            self.assertEqual(r2, expected)
            self.assertEqual(state.summary().calls, 2)
            loaded = torch.compiler.precompile.load_files(**paths)
            z = torch.randn(2, 3)
            expected = (z.clone() + 1) * 2
            self.assertEqual(loaded(z), expected)

    def test_tracer_dynamo_parameter_example_input(self):
        # The frozen snapshot must preserve the input's exact Python type:
        # nn.Parameter disables __torch_function__, so a bare as_strided view
        # decays to plain Tensor and TENSOR_MATCH's pytype check then fails
        # every guard re-check ("does not match any example input").
        p = torch.nn.Parameter(torch.randn(3))
        x = torch.randn(3)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_param_scale,
            example_inputs=[(p, x)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        q = torch.nn.Parameter(torch.randn(3))
        y = torch.randn(3)
        self.assertEqual(loaded(q, y), (q * y).sum())

    def test_tracer_dynamo_grad_reading_step(self):
        # The frozen snapshot must carry the input's .grad by reference: the
        # alias is a fresh leaf whose .grad starts None, so a fn reading a
        # non-None x.grad (an optimizer-style step) would fail its
        # GradSource-rooted guard on every re-check.
        def make_input():
            x = torch.randn(3, requires_grad=True)
            x.sum().backward()
            return x

        x = make_input()
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_grad_step,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        y = make_input()
        self.assertEqual(loaded(y), y - 0.1 * y.grad)
        with tempfile.TemporaryDirectory() as tmp:
            z = make_input()
            [r], state = torch.compiler.precompile.stateful(
                _precompile_dynamo_grad_step,
                example_inputs=[(z,)],
                state=None,
                backend="eager",
                **_stateful_paths(tmp),
            )
            self.addCleanup(state.close)
            self.assertEqual(r, z - 0.1 * z.grad)

    def test_tracer_dynamo_stateful_returns_list_per_example(self):
        # Always a list, one entry per example tuple of the call -- never
        # unwrapped, so a fn that itself returns a list stays unambiguous.
        with tempfile.TemporaryDirectory() as tmp:
            xs = [torch.randn(2, 4), torch.randn(3, 4)]
            results, state = torch.compiler.precompile.stateful(
                _precompile_dynamo_dynamic,
                example_inputs=[(xs[0],), (xs[1],)],
                state=None,
                backend="eager",
                **_stateful_paths(tmp),
            )
            self.addCleanup(state.close)
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 2)
            for result, x in zip(results, xs):
                self.assertEqual(result, _precompile_dynamo_dynamic(x))

    def test_write_dynamo_artifact_files_utf8_and_creates_parent_dirs(self):
        from torch._precompile import _write_dynamo_artifact_files

        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = os.path.join(tmp, "out", "artifact.py")
            cache_path = os.path.join(tmp, "out", "artifact.cache")
            content = "# d\u00e9j\u00e0 vu\n"
            _write_dynamo_artifact_files(content, b"cache", artifact_path, cache_path)
            with open(artifact_path, "rb") as f:
                self.assertEqual(f.read(), content.encode("utf-8"))
            with open(cache_path, "rb") as f:
                self.assertEqual(f.read(), b"cache")
            leftovers = [
                n
                for n in os.listdir(os.path.dirname(artifact_path))
                if n.endswith(".tmp")
            ]
            self.assertEqual(leftovers, [])

    def test_tracer_dynamo_load_degrades_on_missing_cache_file(self):
        # A crash between the two renames of the FIRST stateful rewrite leaves
        # an artifact with no cache file; the pair must still load.
        with tempfile.TemporaryDirectory() as tmp:
            paths = _stateful_paths(tmp)
            x = torch.randn(3)
            _, state = torch.compiler.precompile.stateful(
                _precompile_dynamo_torch_sin,
                example_inputs=[(x,)],
                state=None,
                backend="eager",
                **paths,
            )
            self.addCleanup(state.close)
            os.unlink(paths["cache_path"])
            with self.assertLogs("torch._precompile", level="WARNING") as logs:
                loaded = torch.compiler.precompile.load_files(**paths)
            self.assertTrue(any("found no cache file" in m for m in logs.output))
            self.assertEqual(loaded(x), _precompile_dynamo_torch_sin(x))

    def test_load_rejects_backend_tag_mismatch_on_both_tracers(self):
        # The in-memory pair is strict on both tracers; only load_files degrades
        # (see test_tracer_dynamo_load_rejects_mismatched_code_cache_pair).
        x = torch.randn(3)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_torch_sin,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        blob["backend"] = "inductor"
        buf = io.BytesIO()
        torch.save(blob, buf)
        with self.assertRaisesRegex(PrecompileError, "cache backend"):
            torch.compiler.precompile.load(code, buf.getvalue())
        mfx_code, mfx_cache = torch.compiler.precompile(
            lambda t: t + 1, example_inputs=[(x,)], backend="eager"
        )
        blob = torch.load(io.BytesIO(mfx_cache), weights_only=True)
        blob["backend"] = "inductor"
        buf = io.BytesIO()
        torch.save(blob, buf)
        with self.assertRaisesRegex(PrecompileError, "cache backend"):
            torch.compiler.precompile.load(mfx_code, buf.getvalue())

    def test_tracer_dynamo_stateful_recompile_limit_advice(self):
        # The one-shot advice ("pass a larger recompile_limit") is unactionable
        # on a resumed state, whose limit is fixed at creation.
        with tempfile.TemporaryDirectory() as tmp:
            paths = _stateful_paths(tmp)
            x = torch.randn(4)
            _, state = torch.compiler.precompile.stateful(
                _precompile_dynamo_scalar,
                example_inputs=[(x, 2)],
                state=None,
                recompile_limit=1,
                backend="eager",
                **paths,
            )
            self.addCleanup(state.close)
            with self.assertRaisesRegex(PrecompileError, r"close\(\) this state"):
                torch.compiler.precompile.stateful(
                    _precompile_dynamo_scalar,
                    example_inputs=[(x, 3)],
                    state=state,
                    recompile_limit=1,
                    backend="eager",
                    **paths,
                )

    def test_tracer_dynamo_stateful_unrenderable_rebuild_reports_and_keeps_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _stateful_paths(tmp)
            x = torch.randn(2, 4)
            _, state = torch.compiler.precompile.stateful(
                _precompile_dynamo_dynamic,
                example_inputs=[(x,)],
                state=None,
                backend="eager",
                **paths,
            )
            self.addCleanup(state.close)
            with mock.patch(
                "torch._precompile._build_dynamo_artifact",
                side_effect=PrecompileError("simulated render failure"),
            ):
                with self.assertRaisesRegex(
                    PrecompileError, "can no longer be rendered"
                ):
                    torch.compiler.precompile.stateful(
                        _precompile_dynamo_dynamic,
                        example_inputs=[(torch.randn(3, 4),)],
                        state=state,
                        backend="eager",
                        **paths,
                    )
            # The last successfully written pair stays loadable.
            loaded = torch.compiler.precompile.load_files(**paths)
            self.assertEqual(loaded(x), _precompile_dynamo_dynamic(x))

    def test_tracer_dynamo_unclosed_state_warns_on_gc(self):
        import gc

        from torch._precompile import _teardown_dynamo_capture

        with tempfile.TemporaryDirectory() as tmp:
            _, state = torch.compiler.precompile.stateful(
                _precompile_dynamo_torch_sin,
                example_inputs=[(torch.randn(3),)],
                state=None,
                backend="eager",
                **_stateful_paths(tmp),
            )
            # The state is about to be dropped without close(), so release the
            # session by hand even if the assertions below fail -- otherwise a
            # failure leaks the pinned session into the rest of the suite.
            self.addCleanup(
                _teardown_dynamo_capture,
                state.package,
                state.capture_target,
                state.pgo_state,
                state.backend_fn,
            )
            with self.assertLogs("torch._precompile", level="WARNING") as logs:
                del state
                gc.collect()
            self.assertTrue(any("without close()" in m for m in logs.output))

    def test_load_rejects_tampered_or_garbage_python_code(self):
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_torch_sin,
            example_inputs=[(torch.randn(3),)],
            tracer="dynamo",
            backend="eager",
        )
        tampered = code.replace('TRACER = "dynamo"', 'TRACER = "nope"')
        self.assertNotEqual(tampered, code)
        with self.assertRaisesRegex(PrecompileError, "unsupported TRACER value"):
            torch.compiler.precompile.load(tampered, cache)
        with self.assertRaisesRegex(PrecompileError, "not valid Python"):
            torch.compiler.precompile.load("def (", cache)

    def test_tracer_dynamo_package_serialization_error_is_translated(self):
        from torch._dynamo.exc import PackageError

        with mock.patch(
            "torch._dynamo.package.CompilePackage",
            side_effect=PackageError("simulated package failure"),
        ):
            with self.assertRaisesRegex(
                PrecompileError, "could not serialize the capture"
            ):
                torch.compiler.precompile(
                    _precompile_dynamo_torch_sin,
                    example_inputs=[(torch.randn(3),)],
                    tracer="dynamo",
                    backend="eager",
                )

    def test_tracer_dynamo_load_warns_when_cache_prime_fails(self):
        # A stale/corrupt inner bundle must degrade to JIT with a warning, not
        # fail the load: the bundle is pure acceleration.
        x = torch.randn(3)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_torch_sin, example_inputs=[(x,)], tracer="dynamo"
        )
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        # A non-None bundle so the prime loop actually calls; the raise is
        # mocked because load_cache_artifacts swallows mere corruption itself.
        blob["artifact"] = [b"bundle"]
        buf = io.BytesIO()
        torch.save(blob, buf)
        with mock.patch(
            "torch.compiler.load_cache_artifacts",
            side_effect=RuntimeError("simulated foreign bundle"),
        ):
            with self.assertLogs("torch._precompile", level="WARNING") as logs:
                loaded = torch.compiler.precompile.load(code, buf.getvalue())
        self.assertTrue(any("could not prime the cache" in m for m in logs.output))
        self.assertEqual(loaded(x), _precompile_dynamo_torch_sin(x))

    def test_load_untrusted_warning_precedes_cache_processing(self):
        # Both halves of a load are untrusted executable input, so the trust
        # warning must fire BEFORE any cache processing: with a corrupt cache
        # envelope, the untrusted-input record comes before the envelope
        # degrade record, and the load still serves (the cache is acceleration
        # only).
        x = torch.randn(3)
        code, _cache = torch.compiler.precompile(
            _precompile_dynamo_torch_sin,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        with self.assertLogs("torch._precompile", level="WARNING") as cm:
            loaded = torch.compiler.precompile.load(code, b"corrupt-envelope")
        untrusted = [i for i, m in enumerate(cm.output) if "untrusted" in m.lower()]
        envelope = [i for i, m in enumerate(cm.output) if "envelope" in m]
        self.assertTrue(untrusted, cm.output)
        self.assertTrue(envelope, cm.output)
        self.assertLess(untrusted[0], envelope[0])
        self.assertEqual(loaded(x), _precompile_dynamo_torch_sin(x))

    def test_tracer_dynamo_accepts_interpreter_singleton_inputs(self):
        # (), Ellipsis, and NotImplemented are process-wide singletons that
        # Dynamo value-guards; a helper default holding one must not read as
        # an environment alias of the caller's input.
        x = torch.randn(3)
        for fn, extra in (
            (_precompile_dynamo_calls_empty_default, ()),
            (_precompile_dynamo_calls_ellipsis_default, ...),
        ):
            with self.subTest(extra=extra):
                code, cache = torch.compiler.precompile(
                    fn, example_inputs=[(x, extra)], tracer="dynamo", backend="eager"
                )
                loaded = torch.compiler.precompile.load(code, cache)
                self.assertEqual(loaded(x, extra), fn(x, extra))

    def test_tracer_dynamo_accepts_unused_function_reference_in_input(self):
        # An argument merely carrying a reference to a function the fn loads
        # as a global is not an alias hazard: a USED reference gets an
        # unserializable identity guard and fails loudly; an unused one never
        # influences dispatch.
        x = torch.randn(3)
        box = _PrecompileDynamoActBox(_precompile_dynamo_act)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_calls_act,
            example_inputs=[(x, box)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(x, box), _precompile_dynamo_calls_act(x, box))

    def test_load_warns_on_empty_in_memory_cache(self):
        x = torch.randn(3)
        code, _ = torch.compiler.precompile(
            _precompile_dynamo_torch_sin,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        with self.assertLogs("torch._precompile", level="WARNING") as logs:
            loaded = torch.compiler.precompile.load(code, b"")
        self.assertTrue(any("got an empty cache" in m for m in logs.output))
        self.assertEqual(loaded(x), _precompile_dynamo_torch_sin(x))

    def test_tracer_dynamo_fresh_stateful_recompile_limit_advice(self):
        # A fresh call that hits the limit self-closes and never returns its
        # state, so it must get the plain advice, not "close() this state".
        with tempfile.TemporaryDirectory() as tmp:
            x = torch.randn(4)
            with self.assertRaisesRegex(
                PrecompileError, r"pass a larger recompile_limit"
            ):
                torch.compiler.precompile.stateful(
                    _precompile_dynamo_scalar,
                    example_inputs=[(x, 2), (x, 3)],
                    state=None,
                    recompile_limit=1,
                    backend="eager",
                    **_stateful_paths(tmp),
                )

    @parametrize("backend", ("inductor", "eager"))
    def test_tracer_dynamo_capture_under_no_grad_is_inference_artifact(self, backend):
        # Capture follows the ambient grad mode exactly like torch.compile: under
        # no_grad the graph is an inference graph (no joint forward/backward on
        # inductor), the artifact records the mode, serves without autograd
        # history, and refuses a grad-mode call whose input requires grad (eager
        # would build history the artifact cannot).
        w = torch.randn(4, 4, requires_grad=True)
        x = torch.randn(2, 4)
        with torch.no_grad():
            code, cache = torch.compiler.precompile(
                _precompile_dynamo_matmul,
                example_inputs=[(x, w)],
                tracer="dynamo",
                backend=backend,
            )
        self.assertIn("_DYNAMO_GRAD_ENABLED = False", code)
        self.assertNotIn("_CompiledFunction", code)
        self.assertNotIn("_inner_call_bw", code)
        loaded = torch.compiler.precompile.load(code, cache)
        with torch.no_grad():
            out = loaded(x, w)
        self.assertIsNone(out.grad_fn)
        self.assertEqual(out, _precompile_dynamo_matmul(x, w).detach())
        with self.assertRaisesRegex(PrecompileError, "inference artifact"):
            loaded(x, w)
        # With no input requiring grad, a grad-mode call serves normally.
        plain = torch.randn(4, 4)
        with torch.no_grad():
            code, cache = torch.compiler.precompile(
                _precompile_dynamo_matmul,
                example_inputs=[(x, plain)],
                tracer="dynamo",
                backend=backend,
            )
        out = torch.compiler.precompile.load(code, cache)(x, plain)
        self.assertIsNone(out.grad_fn)
        self.assertEqual(out, _precompile_dynamo_matmul(x, plain))

    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    def test_tracer_dynamo_environment_change_between_stateful_calls_keeps_capturing(
        self,
    ):
        # The guard on len(_PRECOMPILE_HISTORY) is an environment guard, dropped
        # at variant creation; mutating the global between stateful calls cannot
        # poison the state, each variant serves its capture-time specialization
        # (the documented environment contract), and the dropped guard is
        # recorded with the value each variant was specialized to.
        _PRECOMPILE_HISTORY.clear()
        self.addCleanup(_PRECOMPILE_HISTORY.clear)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _stateful_paths(tmp)
            x3 = torch.randn(3)
            [r1], state = torch.compiler.precompile.stateful(
                _precompile_dynamo_reads_history,
                example_inputs=[(x3,)],
                state=None,
                backend="eager",
                **paths,
            )
            self.addCleanup(state.close)
            self.assertEqual(r1, x3 * 1)
            _PRECOMPILE_HISTORY.append(1)
            x4 = torch.randn(4)  # a new static variant, captured with len == 1
            [r2], state = torch.compiler.precompile.stateful(
                _precompile_dynamo_reads_history,
                example_inputs=[(x4,)],
                state=state,
                backend="eager",
                **paths,
            )
            self.assertEqual(r2, x4 * 2)
            summary = state.summary()
            self.assertEqual(summary.variants, 2)
            recorded = {
                value
                for _, source, value in summary.dropped_guards
                if source == "G['_PRECOMPILE_HISTORY']"
            }
            self.assertEqual(recorded, {"[]", "[1]"})
            loaded = torch.compiler.precompile.load_files(**paths)
            self.assertEqual(loaded(x3), x3 * 1)
            self.assertEqual(loaded(x4), x4 * 2)

    @unittest.skipUnless(TEST_CUDA, "measures CUDA allocator growth")
    def test_tracer_dynamo_stateful_does_not_retain_examples(self):
        # The state keeps no example snapshots (the guard filter runs once, at
        # variant creation), so repeated calls on fresh 64 MB inputs must not
        # grow allocated memory by a multiple of the input size.
        import gc

        def batch():
            return torch.randn(16 * 1024 * 1024, device="cuda")

        with tempfile.TemporaryDirectory() as tmp:
            paths = _stateful_paths(tmp)
            _, state = torch.compiler.precompile.stateful(
                _precompile_dynamo_torch_sin,
                example_inputs=[(batch(),)],
                state=None,
                backend="eager",
                **paths,
            )
            self.addCleanup(state.close)
            gc.collect()
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            for _ in range(4):
                _, state = torch.compiler.precompile.stateful(
                    _precompile_dynamo_torch_sin,
                    example_inputs=[(batch(),)],
                    state=state,
                    backend="eager",
                    **paths,
                )
            gc.collect()
            torch.cuda.synchronize()
            self.assertLess(torch.cuda.memory_allocated() - base, 64 * 1024 * 1024)

    def test_tracer_dynamo_rejects_bytecode_reading_user_globals(self):
        # Dynamo's transformed bytecode reconstructs a returned global object and
        # replays side effects on globals by loading them by name; the artifact
        # runs in its own namespace, so such a global is rejected at capture
        # (naming it) instead of raising NameError at serve.
        self.addCleanup(setattr, _PrecompileDynamoCfg, "counter", 0)
        self.addCleanup(_PRECOMPILE_LOG.clear)
        for fn, name in (
            (_precompile_dynamo_returns_global_class, "_PrecompileDynamoCfg"),
            (_precompile_dynamo_mutates_class_attr, "_PrecompileDynamoCfg"),
            (_precompile_dynamo_appends_global_list, "_PRECOMPILE_LOG"),
        ):
            with self.subTest(fn=fn.__name__):
                with self.assertRaisesRegex(
                    PrecompileError, rf"reads the global\(s\) \['{name}'\]"
                ):
                    torch.compiler.precompile(
                        fn,
                        example_inputs=[(torch.randn(3),)],
                        tracer="dynamo",
                        backend="eager",
                    )

    def test_tracer_dynamo_accepts_class_attribute_read_and_library_reference(self):
        # The exact environment check keys on what Dynamo actually read: a class
        # whose METHOD references a tensor, or a large library module (sympy),
        # is fine when the fn only reads a plain attribute; the old object-graph
        # prescan rejected the former and spent seconds walking the latter.
        x = torch.randn(3)
        for fn, expected in (
            (_precompile_dynamo_reads_class_scale, x * 2),
            (_precompile_dynamo_refs_sympy, x + len(sympy.__name__)),
        ):
            with self.subTest(fn=fn.__name__):
                code, cache = torch.compiler.precompile(
                    fn, example_inputs=[(x,)], tracer="dynamo", backend="eager"
                )
                self.assertEqual(
                    torch.compiler.precompile.load(code, cache)(x), expected
                )

    def test_tracer_dynamo_serves_across_num_threads_change(self):
        # GLOBAL_STATE records the capture machine's thread count; the artifact
        # keeps that field in sync with the serving process (at load and after a
        # later change), so thread count never fails dispatch.
        x = torch.randn(3)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_torch_sin,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        n = torch.get_num_threads()
        if n < 2:
            self.skipTest("needs at least two threads to change the count")
        self.addCleanup(torch.set_num_threads, n)
        torch.set_num_threads(n - 1)
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(x), torch.sin(x))
        torch.set_num_threads(n)  # changed again after load
        self.assertEqual(loaded(x), torch.sin(x))

    def test_tracer_dynamo_ambient_state_mismatch_names_global_state(self):
        # Autocast is part of the per-call GLOBAL_STATE check (unlike num_threads);
        # a miss under autocast names the differing state rather than advising
        # to add an example.
        x = torch.randn(3)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_torch_sin,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, r"autocast.*GLOBAL_STATE"):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loaded(x)

    def test_tracer_dynamo_dispatch_miss_names_failing_guard(self):
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_scalar,
            example_inputs=[(x, 2)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(
            PrecompileError, r"no captured Dynamo variant.*L\['x'\].*dtype"
        ):
            loaded(x.double(), 2)

    def test_tracer_dynamo_unsupported_construct_is_not_reported_as_graph_break(self):
        with self.assertRaisesRegex(PrecompileError, "Data-dependent branching") as cm:
            torch.compiler.precompile(
                _precompile_dynamo_data_dependent,
                example_inputs=[(torch.randn(3),)],
                tracer="dynamo",
                backend="eager",
            )
        self.assertNotIn("does not support graph breaks", str(cm.exception))

    @parametrize("backend", ("inductor", "eager"))
    def test_tracer_dynamo_no_grad_view_of_input_matches_eager(self, backend):
        # Under an ambient no_grad, an output that is a view of an input is
        # regenerated as a no_grad view of that input (base, requires_grad), so a
        # later in-place write under grad mode raises exactly as eager does
        # instead of silently mutating the requires_grad leaf.
        x = torch.randn(4, requires_grad=True)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_first_element,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend=backend,
        )
        loaded = torch.compiler.precompile.load(code, cache)
        with torch.no_grad():
            served = loaded(x)
            ref = _precompile_dynamo_first_element(x)
        self.assertEqual(served, ref)
        self.assertTrue(served._is_view())
        self.assertIs(served._base, x)
        self.assertEqual(served.requires_grad, ref.requires_grad)
        with self.assertRaisesRegex(RuntimeError, "created in no_grad mode"):
            served.add_(1.0)

    def test_tracer_dynamo_mutates_inputs_flag(self):
        # The driver checks runtime inputs for storage overlap only when a captured
        # graph mutates an input: the flag comes from the FX graph on eager and
        # from AOTAutograd's input metadata on inductor training graphs.
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_scalar,
            example_inputs=[(x, 2)],
            tracer="dynamo",
            backend="eager",
        )
        self.assertIn("_DYNAMO_MUTATES_INPUTS = False", code)
        code, _ = torch.compiler.precompile(
            _precompile_dynamo_inplace_step,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        self.assertIn("_DYNAMO_MUTATES_INPUTS = True", code)
        code, _ = torch.compiler.precompile(
            _precompile_dynamo_inplace_add,
            example_inputs=[(torch.randn(4), torch.randn(4, requires_grad=True))],
            tracer="dynamo",
        )
        self.assertIn("_DYNAMO_MUTATES_INPUTS = True", code)

    @parametrize("tracer", ("make_fx", "dynamo"))
    def test_example_inputs_generator_rejected(self, tracer):
        x = torch.randn(3)
        with self.assertRaisesRegex(TypeError, "sequence .* got generator"):
            torch.compiler.precompile(
                _precompile_dynamo_torch_sin,
                example_inputs=((x,) for _ in range(1)),
                tracer=tracer,
                backend="eager",
            )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(TypeError, "sequence .* got generator"):
                torch.compiler.precompile.stateful(
                    _precompile_dynamo_torch_sin,
                    example_inputs=((x,) for _ in range(1)),
                    backend="eager",
                    **_stateful_paths(tmp),
                )
            self.assertEqual(os.listdir(tmp), [])

    def test_tracer_dynamo_state_is_context_manager(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, state = torch.compiler.precompile.stateful(
                _precompile_dynamo_torch_sin,
                example_inputs=[(torch.randn(3),)],
                state=None,
                backend="eager",
                **_stateful_paths(tmp),
            )
            self.addCleanup(state.close)  # idempotent; the with-block closes below
            self.assertIsInstance(state, torch.compiler.precompile.PrecompileState)
            self.assertIsInstance(
                state.summary(), torch.compiler.precompile.PrecompileStateSummary
            )
            with state as entered:
                self.assertIs(entered, state)
            self.assertTrue(state.closed)

    def test_make_fx_artifact_rejects_keyword_call(self):
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(2, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)], backend="eager"
        )
        loaded = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "positional arguments only"):
            loaded(m, t=x)

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_capture_ignores_global_pgo_dynamic_record(self):
        # torch.compile marks a dim dynamic in the PROCESS-GLOBAL PGO record;
        # precompile drives an ISOLATED PGO record, so a one-example capture must
        # still produce a static artifact -- identical to a clean capture -- rather
        # than inheriting the global automatic-dynamic decision (which would make a
        # single example compile a dynamic graph).
        from torch._dynamo.testing import CompileCounter

        def counts(code):
            return [
                line
                for line in code.splitlines()
                if line.startswith(
                    ("VARIANT_COUNT ", "GRAPH_COUNT ", "DYNAMIC_GRAPH_COUNT ")
                )
            ]

        fn = _precompile_dynamo_dynamic
        x = torch.randn(2, 4)
        # A clean capture (isolated PGO) of one example is a single static variant.
        clean_code, _ = torch.compiler.precompile(
            fn, example_inputs=[(x,)], tracer="dynamo", backend="eager"
        )
        self.assertIn("DYNAMIC_GRAPH_COUNT = 0", clean_code)

        # Pollute the global record: automatic dynamic folds sizes 3 and 5 into one
        # dynamic recompile, after which a fresh size does not recompile -- proving
        # the global record now holds a dynamic dim 0 for this code.
        torch._dynamo.reset()
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter)
        for size in (2, 3, 5):
            compiled(torch.randn(size, 4))
        self.assertEqual(counter.frame_count, 2)
        compiled(torch.randn(9, 4))
        self.assertEqual(counter.frame_count, 2)  # global dim 0 is dynamic now

        # The isolated capture ignores that: one example is still static, and the
        # variant/graph set matches the clean capture.
        polluted_code, cache = torch.compiler.precompile(
            fn, example_inputs=[(x,)], tracer="dynamo", backend="eager"
        )
        self.assertIn("DYNAMIC_GRAPH_COUNT = 0", polluted_code)
        self.assertEqual(counts(polluted_code), counts(clean_code))
        self.assertEqual(torch.compiler.precompile.load(polluted_code, cache)(x), fn(x))

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_capture_leaves_global_pgo_state_untouched(self):
        from torch._dynamo import pgo

        # Pollute the global record first so there is real content to (not) mutate.
        fn = _precompile_dynamo_dynamic
        torch._dynamo.reset()
        compiled = torch.compile(fn, backend="eager")
        for size in (2, 3):
            compiled(torch.randn(size, 4))
        # Snapshot BY VALUE (repr captures the automatic_dynamic contents, not just
        # the key set): a capture that leaked into the global record would record a
        # new size onto the existing entry and change its repr.
        before = {str(k): repr(v) for k, v in pgo.get_code_state().items()}
        self.assertTrue(before)  # the pollution recorded something
        torch.compiler.precompile(
            fn,
            example_inputs=[(torch.randn(5, 4),)],
            tracer="dynamo",
            backend="eager",
        )
        after = {str(k): repr(v) for k, v in pgo.get_code_state().items()}
        self.assertEqual(before, after)  # capture ran on an isolated PGO record


@skipIfTorchDynamo("precompile's make_fx capture is incompatible with dynamo wrapping")
class TestPrecompileNumerics(TestCase):
    # Numeric-correctness tests run device-generically so the same coverage
    # exercises the CUDA lowering, not just CPU.

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_dynamic_numerics(self, device):
        from torch._inductor.utils import fresh_cache

        examples = [
            (make_tensor((size, 4), device=device, dtype=torch.float32),)
            for size in (2, 3, 5)
        ]
        with fresh_cache():
            code, cache = torch.compiler.precompile(
                _precompile_dynamo_dynamic,
                example_inputs=examples,
                tracer="dynamo",
            )

        self.assertIn("DYNAMIC_GRAPH_COUNT = 1", code)
        if torch.device(device).type in ("cuda", "xpu"):
            self.assertIn("@triton.jit", code)  # GPU lowering inlines Triton source
        for _, loaded in _default_and_inlined_loaders(code, cache, "inductor"):
            for size in (2, 7):
                x = make_tensor((size, 4), device=device, dtype=torch.float32)
                self.assertEqual(loaded(x), _precompile_dynamo_dynamic(x))

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_training_numerics(self, device):
        from torch._inductor.utils import fresh_cache

        examples = [
            (
                make_tensor(
                    (size, 4), device=device, dtype=torch.float32, requires_grad=True
                ),
            )
            for size in (2, 3, 5)
        ]
        with fresh_cache():
            code, cache = torch.compiler.precompile(
                _precompile_dynamo_dynamic,
                example_inputs=examples,
                tracer="dynamo",
            )

        self.assertIn("DYNAMIC_GRAPH_COUNT = 1", code)
        self.assertIn("_inner_call_bw", code)
        if torch.device(device).type in ("cuda", "xpu"):
            self.assertIn("@triton.jit", code)
        for _, loaded in _default_and_inlined_loaders(code, cache, "inductor"):
            x = make_tensor(
                (7, 4), device=device, dtype=torch.float32, requires_grad=True
            )
            ref = x.detach().clone().requires_grad_()
            expected = _precompile_dynamo_dynamic(ref)
            expected.sum().backward()
            actual = loaded(x)
            self.assertTrue(actual.requires_grad)
            actual.sum().backward()
            self.assertEqual(actual, expected)
            self.assertEqual(x.grad, ref.grad)

    # The undefined-tangent (backward mask) specialization is an inductor-backward
    # feature that also runs on CUDA, so these numeric checks are device-generic
    # rather than CPU-only.
    def test_tracer_dynamo_unseen_tangent_pattern_uses_default_backward(self, device):
        # A forward-only capture observes no partial backward; an unseen
        # output-tangent pattern at serve time falls back to the always-covered
        # all-tangents-defined backward (materializing the missing tangent), so
        # the gradients match eager instead of raising.
        x = make_tensor((4,), device=device, dtype=torch.float32, requires_grad=True)
        y = make_tensor((4,), device=device, dtype=torch.float32, requires_grad=True)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_independent_outputs,
            example_inputs=[(x, y)],
            tracer="dynamo",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        loaded(x, y)[0].sum().backward()  # only output 0 gets a tangent
        self.assertEqual(x.grad, x.detach().cos())
        self.assertIsNone(y.grad)

    def test_tracer_dynamo_nondifferentiable_output_backward(self, device):
        # A non-differentiable output's tangent is ALWAYS undefined, so keying
        # variants on the raw scanned mask (rather than the canonical mask over
        # specializable outputs, like the live runtime) once made the
        # auto-covered mask 0 unreachable and every backward raised. The ordinary
        # backward must still serve and match eager.
        x = make_tensor((4,), device=device, dtype=torch.float32, requires_grad=True)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_nondiff_second_output,
            example_inputs=[(x,)],
            tracer="dynamo",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        y = make_tensor((4,), device=device, dtype=torch.float32, requires_grad=True)
        out_a, out_b = loaded(y)
        self.assertIsNone(out_b.grad_fn)
        out_a.sum().backward()
        self.assertEqual(y.grad, y.detach().cos())

    def test_tracer_dynamo_inductor_double_backward_raises(self, device):
        # The inductor backend precompiles only the first-order backward; the
        # emitted _DoubleBackward bridge must raise the documented error on a
        # second-order backward instead of silently producing wrong grads.
        x = make_tensor((4,), device=device, dtype=torch.float32, requires_grad=True)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_torch_sin,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="inductor",
        )
        out = torch.compiler.precompile.load(code, cache)(x)
        (grad,) = torch.autograd.grad(out.sum(), x, create_graph=True)
        with self.assertRaisesRegex(
            RuntimeError, "does not currently support double backward"
        ):
            grad.sum().backward()

    @parametrize("backend", ("inductor", "eager"))
    def test_tracer_dynamo_inplace_mutation_flips_nograd_input(self, device, backend):
        # In-place mutation of a no-grad input from a differentiable source
        # (x.add_(y_requires_grad)) flips the LIVE input's requires_grad while the
        # example runs. The recorded snapshot must keep the ENTRY state the guards
        # recorded, and the served grads must match eager, on both backends.
        for fn, expected_grad in (
            (_precompile_dynamo_inplace_add, torch.ones(4, device=device)),
            (_precompile_dynamo_inplace_copy, torch.full((4,), 2.0, device=device)),
        ):
            with self.subTest(fn=fn.__name__):
                x0 = make_tensor((4,), device=device, dtype=torch.float32)
                y0 = make_tensor(
                    (4,), device=device, dtype=torch.float32, requires_grad=True
                )
                code, cache = torch.compiler.precompile(
                    fn, example_inputs=[(x0, y0)], tracer="dynamo", backend=backend
                )
                loaded = torch.compiler.precompile.load(code, cache)
                p = make_tensor((4,), device=device, dtype=torch.float32)
                q = make_tensor(
                    (4,), device=device, dtype=torch.float32, requires_grad=True
                )
                p_ref = p.detach().clone()
                q_ref = q.detach().clone().requires_grad_()
                out = loaded(p, q)
                expected = fn(p_ref, q_ref)
                out.sum().backward()
                expected.sum().backward()
                self.assertEqual(out, expected)
                self.assertEqual(p, p_ref)  # the mutation lands on the input
                self.assertEqual(q.grad, expected_grad)
                self.assertEqual(q.grad, q_ref.grad)

    def test_tracer_dynamo_stateful_partial_backward_keeps_default_mask(self, device):
        # A partial backward between calls records a nonzero tangent mask; the
        # next rewrite must still cover the ordinary all-defined backward (mask 0)
        # and the observed partial pattern, matching eager grads.
        def clones():
            a = make_tensor(
                (4,), device=device, dtype=torch.float32, requires_grad=True
            )
            b = make_tensor(
                (4,), device=device, dtype=torch.float32, requires_grad=True
            )
            return a, b

        with tempfile.TemporaryDirectory() as tmp:
            paths = _stateful_paths(tmp)
            x, y = clones()
            [(out_a, _out_b)], state = torch.compiler.precompile.stateful(
                _precompile_dynamo_independent_outputs,
                example_inputs=[(x, y)],
                state=None,
                **paths,
            )
            self.addCleanup(state.close)
            out_a.sum().backward()  # partial: only output 0 gets a tangent
            _, state = torch.compiler.precompile.stateful(
                _precompile_dynamo_independent_outputs,
                example_inputs=[clones()],
                state=state,
                **paths,
            )
            code, cache = _read_pair(paths)
            loaded = torch.compiler.precompile.load(code, cache)
            p, q = clones()
            served_a, served_b = loaded(p, q)
            (served_a.sum() + served_b.sum()).backward()  # all-defined backward
            self.assertEqual(p.grad, p.detach().cos())
            self.assertEqual(q.grad, -q.detach().sin())
            p2, q2 = clones()
            loaded(p2, q2)[0].sum().backward()  # the observed partial pattern
            self.assertEqual(p2.grad, p2.detach().cos())

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_stateful_partial_mask_survives_dynamic_recompile(
        self, device
    ):
        # A tangent mask observed on a backend must survive a later
        # automatic-dynamic recompile: the new dynamic backend supersedes the
        # static one in newest-first dispatch, so it serves the sizes the static
        # backend was captured for and must cover the masks that backend observed
        # -- otherwise the exact captured backward pattern raises after recompile.
        def clones(size):
            a = make_tensor(
                (size,), device=device, dtype=torch.float32, requires_grad=True
            )
            b = make_tensor(
                (size,), device=device, dtype=torch.float32, requires_grad=True
            )
            return a, b

        with tempfile.TemporaryDirectory() as tmp:
            paths = _stateful_paths(tmp)
            x, y = clones(4)
            [(out_a, _out_b)], state = torch.compiler.precompile.stateful(
                _precompile_dynamo_independent_outputs,
                example_inputs=[(x, y)],
                state=None,
                **paths,
            )
            self.addCleanup(state.close)
            out_a.sum().backward()  # partial mask observed on the size-4 backend
            _, state = torch.compiler.precompile.stateful(
                _precompile_dynamo_independent_outputs,
                example_inputs=[clones(8)],
                state=state,
                **paths,
            )
            code, cache = _read_pair(paths)
            loaded = torch.compiler.precompile.load(code, cache)
            p, q = clones(4)  # served by the newer dynamic variant
            loaded(p, q)[0].sum().backward()
            self.assertEqual(p.grad, p.detach().cos())
            self.assertIsNone(q.grad)

    def test_tracer_dynamo_cache_bundle_is_populated(self, device):
        # Every dynamo inductor artifact -- inference and differentiable -- ships
        # a real per-graph cache bundle that load() feeds to load_cache_artifacts
        # (the bundle used to be [None] and primed nothing).
        for requires_grad in (False, True):
            with self.subTest(requires_grad=requires_grad):
                x = make_tensor(
                    (4, 4),
                    device=device,
                    dtype=torch.float32,
                    requires_grad=requires_grad,
                )
                code, cache = torch.compiler.precompile(
                    _precompile_dynamo_torch_sin, example_inputs=[(x,)], tracer="dynamo"
                )
                blob = torch.load(io.BytesIO(cache), weights_only=True)
                self.assertEqual(len(blob["artifact"]), 1)
                self.assertIsInstance(blob["artifact"][0], bytes)
                primed = []
                original = torch.compiler.load_cache_artifacts

                def spy(bundle):
                    info = original(bundle)
                    primed.append(info)
                    return info

                with mock.patch("torch.compiler.load_cache_artifacts", spy):
                    loaded = torch.compiler.precompile.load(code, cache)
                self.assertEqual(len(primed), 1)
                self.assertTrue(primed[0].inductor_artifacts)
                self.assertEqual(loaded(x), torch.sin(x))

    def test_plain_function(self, device):
        def f(x, y):
            return (x @ y).sin(), x + y

        a = make_tensor((4, 4), device=device, dtype=torch.float32)
        b = make_tensor((4, 4), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(f, example_inputs=[(a, b)])
        self.assertIsInstance(code, str)
        self.assertIsInstance(cache, bytes)

        f_c = torch.compiler.precompile.load(code, cache)
        out = f_c(a, b)
        ref = f(a, b)
        self.assertEqual(out[0], ref[0])
        self.assertEqual(out[1], ref[1])

    def test_module_params_and_buffers_are_lifted(self, device):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)
                self.register_buffer("b2", torch.randn(3))

            def forward(self, x):
                return torch.relu(self.lin(x)) + self.b2

        m = M().to(device).eval()
        x = make_tensor((5, 4), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_multiple_module_args(self, device):
        # More than one nn.Module arg: each module's params are lifted with
        # m{i}.-prefixed names. Both modules are passed again at runtime.
        a = torch.nn.Linear(4, 4).to(device).eval()
        b = torch.nn.Linear(4, 3).to(device).eval()
        x = make_tensor((2, 4), device=device, dtype=torch.float32)
        ref = b(torch.relu(a(x)))

        code, cache = torch.compiler.precompile(
            lambda ma, mb, x: mb(torch.relu(ma(x))), example_inputs=[(a, b, x)]
        )
        self.assertIn(
            "PARAM_NAMES = ['m0.weight', 'm0.bias', 'm1.weight', 'm1.bias']", code
        )

        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(a, b, x), ref)

    def test_inplace_on_intermediate_is_allowed(self, device):
        # In-place ops on intermediates (e.g. nn.ReLU(inplace=True)) are fine -- they
        # do not touch any input -- and must NOT be rejected as input mutation.
        m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(inplace=True))
        m.to(device).eval()
        x = make_tensor((5, 4), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_training_backward_harvest_matches_eager(self, device):
        # A training step that calls loss.backward(): precompile scatters the
        # parameter grads onto the runtime model's .grad fields (mirroring eager
        # .backward()) and returns fn's own result (None here).
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        ).to(device)
        loss_fn = torch.nn.MSELoss()
        # Keep magnitudes small (make_tensor defaults to a wide range) so the SGD
        # loop below converges rather than diverges.
        x = make_tensor((5, 4), device=device, dtype=torch.float32, low=-1, high=1)
        target = make_tensor((5, 3), device=device, dtype=torch.float32, low=-1, high=1)

        ref = copy.deepcopy(model)
        loss_fn(ref(x), target).backward()
        ref_grads = [p.grad.clone() for p in ref.parameters()]

        def train_step(model, x, target):
            loss_fn(model(x), target).backward()

        code, cache = torch.compiler.precompile(
            train_step, example_inputs=[(model, x, target)]
        )
        f_c = torch.compiler.precompile.load(code, cache)

        # The model is passed at runtime (no weights baked); the artifact mutates
        # model.parameters().grad in place, returning fn's result (None).
        out = f_c(model, x, target)
        self.assertIsNone(out)
        for p, rg in zip(model.parameters(), ref_grads):
            self.assertEqual(p.grad, rg)

        # Grads accumulate like eager: a second call without zeroing doubles them.
        f_c(model, x, target)
        for p, rg in zip(model.parameters(), ref_grads):
            self.assertEqual(p.grad, rg * 2)

        # A standard zero_grad / step loop reduces loss.
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        losses = []
        for _ in range(5):
            opt.zero_grad()
            f_c(model, x, target)
            losses.append(loss_fn(model(x), target).item())
            opt.step()
        self.assertLess(losses[-1], losses[0])

    def test_frozen_params_grad_matches_eager(self, device):
        # Params that do not receive a gradient -- a frozen (requires_grad=False)
        # backbone, or a param that does not contribute to the loss -- must keep
        # .grad = None after the step, exactly like eager .backward(). precompile must
        # NOT zero-fill them (regression test for the old all-params zero-fill).
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        ).to(device)
        for p in model[0].parameters():
            p.requires_grad_(False)  # freeze the first linear
        loss_fn = torch.nn.MSELoss()
        x = make_tensor((5, 4), device=device, dtype=torch.float32, low=-1, high=1)
        target = make_tensor((5, 3), device=device, dtype=torch.float32, low=-1, high=1)

        ref = copy.deepcopy(model)
        loss_fn(ref(x), target).backward()

        def train_step(model, x, target):
            loss_fn(model(x), target).backward()

        code, cache = torch.compiler.precompile(
            train_step, example_inputs=[(model, x, target)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        f_c(model, x, target)
        for (n, p), (_, rp) in zip(model.named_parameters(), ref.named_parameters()):
            if rp.grad is None:
                self.assertIsNone(p.grad, f"{n}: expected no grad, matching eager")
            else:
                self.assertEqual(p.grad, rp.grad)

    def test_multiple_modules_backward_grad_scatter(self, device):
        # Two distinct module args + a backward: grads must scatter onto the correct
        # module's params via the cross-module GRAD_PARAM_INDICES mapping. One module
        # is partly frozen so the test also pins the index shift across modules.
        torch.manual_seed(0)
        a = torch.nn.Linear(4, 4).to(device)
        b = torch.nn.Linear(4, 3).to(device)
        a.bias.requires_grad_(False)  # a frozen param shifts later indices
        loss_fn = torch.nn.MSELoss()
        x = make_tensor((5, 4), device=device, dtype=torch.float32, low=-1, high=1)
        target = make_tensor((5, 3), device=device, dtype=torch.float32, low=-1, high=1)

        ref_a, ref_b = copy.deepcopy(a), copy.deepcopy(b)
        loss_fn(ref_b(torch.relu(ref_a(x))), target).backward()

        def train_step(ma, mb, x, target):
            loss_fn(mb(torch.relu(ma(x))), target).backward()

        code, cache = torch.compiler.precompile(
            train_step, example_inputs=[(a, b, x, target)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        f_c(a, b, x, target)
        for (n, p), (_, rp) in zip(a.named_parameters(), ref_a.named_parameters()):
            if rp.grad is None:
                self.assertIsNone(p.grad, f"a.{n}: expected no grad")
            else:
                self.assertEqual(p.grad, rp.grad, f"a.{n}")
        for (n, p), (_, rp) in zip(b.named_parameters(), ref_b.named_parameters()):
            self.assertEqual(p.grad, rp.grad, f"b.{n}")

    def test_tied_weights_lifted_once(self, device):
        # A tied weight (same tensor under multiple names) must become a single
        # lifted input: otherwise it is double-counted (double optimizer step) and
        # gradients are split rather than accumulated.
        class Tied(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Linear(4, 4, bias=False)
                self.b = torch.nn.Linear(4, 4, bias=False)
                self.b.weight = self.a.weight  # tie

            def forward(self, x):
                return self.b(torch.relu(self.a(x)))

        torch.manual_seed(0)
        m = Tied().to(device)
        x = make_tensor((3, 4), device=device, dtype=torch.float32)

        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, x), m(x))
        # The tied weight is lifted once (single name), so it is one graph input.
        self.assertIn("PARAM_NAMES = ['a.weight']", code)

        # Training scatters a single grad onto the shared weight, matching eager's
        # accumulation into the tied parameter.
        ref = copy.deepcopy(m)
        ref(x).sum().backward()
        ref_grad = ref.a.weight.grad

        code, cache = torch.compiler.precompile(
            lambda model, x: model(x).sum().backward(), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        f_c(m, x)
        self.assertEqual(m.a.weight.grad, ref_grad)
        # The tie means a.weight and b.weight are the same object, so b sees it too.
        self.assertIs(m.a.weight.grad, m.b.weight.grad)

    def test_backend_eager_plain_function(self, device):
        # backend="eager" runs the captured graph as-is and matches eager.
        def f(x, y):
            return (x @ y).sin(), x + y

        a = make_tensor((4, 4), device=device, dtype=torch.float32)
        b = make_tensor((4, 4), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(
            f, example_inputs=[(a, b)], backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        out = f_c(a, b)
        ref = f(a, b)
        self.assertEqual(out[0], ref[0])
        self.assertEqual(out[1], ref[1])

    def test_backend_eager_module(self, device):
        m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())
        m.to(device).eval()
        x = make_tensor((5, 4), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)], backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_backend_eager_training_harvest(self, device):
        # The backward-harvest contract holds for the eager backend too.
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        ).to(device)
        loss_fn = torch.nn.MSELoss()
        x = make_tensor((5, 4), device=device, dtype=torch.float32, low=-1, high=1)
        target = make_tensor((5, 3), device=device, dtype=torch.float32, low=-1, high=1)

        ref = copy.deepcopy(model)
        loss_fn(ref(x), target).backward()
        ref_grads = [p.grad.clone() for p in ref.parameters()]

        def train_step(model, x, target):
            loss_fn(model(x), target).backward()

        code, cache = torch.compiler.precompile(
            train_step, example_inputs=[(model, x, target)], backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        out = f_c(model, x, target)
        self.assertIsNone(out)
        for p, rg in zip(model.parameters(), ref_grads):
            self.assertEqual(p.grad, rg)

    def test_backend_eager_batchnorm(self, device):
        # The captured graph bakes a ``device`` constant (BatchNorm's
        # num_batches_tracked path), one of fx's custom builtins. The eager
        # standalone source must inject the full custom-builtin set, else this
        # raises NameError: name 'device' is not defined.
        def fresh():
            torch.manual_seed(0)
            m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
            m.train()
            return m.to(device)

        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        ref = fresh()
        ref_out = ref(x)
        ref_rm = ref[1].running_mean.clone()

        code, cache = torch.compiler.precompile(
            lambda m, xx: m(xx), example_inputs=[(fresh(), x)], backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        run = fresh()
        self.assertEqual(f_c(run, x), ref_out)
        self.assertEqual(run[1].running_mean, ref_rm)

    def test_backend_eager_inf_constant(self, device):
        # masked_fill to -inf bakes a bare ``inf`` token into gm.code (another fx
        # custom builtin); the eager standalone source must provide it.
        def f(x):
            return torch.relu(x).masked_fill(x < 0, float("-inf"))

        x = make_tensor((8,), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(
            f, example_inputs=[(x,)], backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(x), f(x))

    def test_batchnorm_train_with_backward(self, device):
        # Training a model containing BatchNorm exercises buffer mutation (running
        # stats) and grad harvest together; grads and running stats must match eager.
        # Inductor fuses the BN backward, so rely on assertEqual's tolerance.
        def fresh():
            torch.manual_seed(0)
            m = torch.nn.Sequential(
                torch.nn.Linear(4, 8), torch.nn.BatchNorm1d(8), torch.nn.Linear(8, 3)
            )
            m.train()
            return m.to(device)

        loss_fn = torch.nn.MSELoss()
        x = make_tensor((16, 4), device=device, dtype=torch.float32, low=-1, high=1)
        target = make_tensor(
            (16, 3), device=device, dtype=torch.float32, low=-1, high=1
        )

        ref = fresh()
        loss_fn(ref(x), target).backward()
        ref_grads = [p.grad.clone() for p in ref.parameters()]
        ref_rm = ref[1].running_mean.clone()

        def train_step(model, x, target):
            loss_fn(model(x), target).backward()

        code, cache = torch.compiler.precompile(
            train_step, example_inputs=[(fresh(), x, target)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        run = fresh()
        f_c(run, x, target)
        for p, rg in zip(run.parameters(), ref_grads):
            self.assertEqual(p.grad, rg)
        self.assertEqual(run[1].running_mean, ref_rm)

    def test_output_alias_supported(self, device):
        # An output that is a view of an input goes through AOTAutograd's output-
        # alias epilogue; precompile reproduces it.
        x = make_tensor((2, 3), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(lambda a: a.t(), example_inputs=[(x,)])
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(x), x.t())

    def test_input_mutation_supported(self, device):
        # In-place input mutation is reflected on the passed tensor (and matches
        # eager), via AOTAutograd's mutation handling composed into the artifact.
        scratch = make_tensor((4,), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(
            lambda a: a.add_(1.0), example_inputs=[(scratch,)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        x = torch.zeros(4, device=device)
        out = f_c(x)
        self.assertEqual(x, torch.ones(4, device=device))
        self.assertEqual(out, torch.ones(4, device=device))

    @unittest.skipUnless(TEST_CUDA, "functionalize_rng_ops seeds via CUDA rng state")
    def test_functionalized_rng_supported(self, device):
        # Functionalized RNG (dropout) threads seed/offset; the AOT backend composes
        # the RNG wrapper in. The artifact runs and produces a valid dropout mask. Even
        # for a CPU tensor the wrapper seeds from CUDARngStateHelper.get_torch_state_as_tuple,
        # which raises unless CUDA is available, so the whole test is gated on TEST_CUDA
        # rather than on the tensor's device.
        import torch._functorch.config as functorch_config

        x = make_tensor((64,), device=device, dtype=torch.float32)
        with functorch_config.patch(functionalize_rng_ops=True):
            code, cache = torch.compiler.precompile(
                lambda a: torch.nn.functional.dropout(a, 0.5, training=True),
                example_inputs=[(x,)],
            )
            f_c = torch.compiler.precompile.load(code, cache)
            out = f_c(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue((out == 0).any())

    def test_batchnorm_train_buffer_mutation(self, device):
        # A stateful module (BatchNorm in training mode) mutates its running stats.
        # precompile reflects that onto the runtime model's buffers and matches eager
        # -- the mutation handling comes from AOTAutograd's codegen.
        def fresh():
            torch.manual_seed(0)
            m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
            m.train()
            return m.to(device)

        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(
            lambda model, xx: model(xx), example_inputs=[(fresh(), x)]
        )

        ref = fresh()
        ref_out = ref(x)
        ref_rm = ref[1].running_mean.clone()
        ref_rv = ref[1].running_var.clone()
        ref_nbt = ref[1].num_batches_tracked.clone()

        f_c = torch.compiler.precompile.load(code, cache)
        run = fresh()
        out = f_c(run, x)
        self.assertEqual(out, ref_out)
        self.assertEqual(run[1].running_mean, ref_rm)
        self.assertEqual(run[1].running_var, ref_rv)
        self.assertEqual(run[1].num_batches_tracked, ref_nbt)

    def test_mutated_duplicate_input(self, device):
        # The same tensor passed twice with a mutation: make_fx resolves the aliasing
        # at trace time (the graph mutates one input and reuses the result), so the
        # artifact reproduces eager when run with the same aliasing. Storage-aliased
        # mutated inputs go through AOTAutograd's now-codegen'd synthetic-base wrapper.
        fn = lambda a, b: (a.mul_(2.0), a + b)[1]  # noqa: E731
        t = make_tensor((4,), device=device, dtype=torch.float32)
        # Clone references BEFORE precompile: capture runs fn once, mutating t.
        ref = t.clone()
        ref_out = fn(ref, ref)
        run = t.clone()

        code, cache = torch.compiler.precompile(fn, example_inputs=[(t, t)])
        f_c = torch.compiler.precompile.load(code, cache)
        out = f_c(run, run)
        self.assertEqual(out, ref_out)

    def test_dynamic_shapes_runs_across_sizes(self, device):
        # An UNBACKED-dynamic batch dim (opted in via mark_unbacked on the input): one
        # artifact runs on many runtime batch sizes (cached AND inlined paths), matching
        # eager. Device-generic so the CUDA unbacked-symint lowering is exercised.
        m = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        )
        m.to(device).eval()
        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        mark_unbacked(x, 0)
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)]
        )
        self.assertIn("USER_INPUT_SHAPES = [(None, 4)]", code)  # dim 0 dynamic
        f_c = torch.compiler.precompile.load(code, cache)
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        blob["artifact"] = None
        buf = io.BytesIO()
        torch.save(blob, buf)
        f_i = torch.compiler.precompile.load(code, buf.getvalue())
        for bs in (8, 16, 1):
            xt = make_tensor((bs, 4), device=device, dtype=torch.float32)
            self.assertEqual(f_c(m, xt), m(xt))  # cached path
            self.assertEqual(f_i(m, xt), m(xt))  # inlined path

    def test_dynamic_shapes_training_across_sizes(self, device):
        # Training (backward) with a dynamic batch; harvested grads match eager across
        # sizes (loss is output.sum() so no cross-input dim-equality guard is needed).
        # Device-generic so the CUDA unbacked-symint backward lowering is exercised.
        torch.manual_seed(0)
        m = torch.nn.Linear(4, 3).to(device)
        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        mark_unbacked(x, 0)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t).sum().backward(), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        for bs in (8, 16, 5):
            run = torch.nn.Linear(4, 3).to(device)
            run.load_state_dict(m.state_dict())
            ref = torch.nn.Linear(4, 3).to(device)
            ref.load_state_dict(m.state_dict())
            xt = make_tensor((bs, 4), device=device, dtype=torch.float32)
            f_c(run, xt)
            ref(xt).sum().backward()
            self.assertEqual(run.weight.grad, ref.weight.grad)

    def test_dynamic_shapes_shared_shape_id(self, device):
        # Two inputs whose batch dims share a shape_id reuse ONE unbacked symbol, so a
        # cross-input matched-batch op (here an add) traces with no dim-equality guard and
        # runs across sizes. Device-generic so the CUDA lowering is exercised.
        m = torch.nn.Linear(4, 4).to(device).eval()
        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        y = make_tensor((8, 4), device=device, dtype=torch.float32)
        mark_unbacked(x, 0, shape_id="b")
        mark_unbacked(y, 0, shape_id="b")
        code, cache = torch.compiler.precompile(
            lambda mm, a, b: mm(a) + b, example_inputs=[(m, x, y)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        for bs in (8, 16, 3):
            xt = make_tensor((bs, 4), device=device, dtype=torch.float32)
            yt = make_tensor((bs, 4), device=device, dtype=torch.float32)
            self.assertEqual(f_c(m, xt, yt), m(xt) + yt)

    def test_mark_unbacked_strict_honored(self, device):
        # mark_unbacked(x, 0, strict=True) is HONORED: the dim is captured as an unbacked
        # symint, so USER_INPUT_SHAPES records None for it and the single artifact runs
        # across runtime sizes, matching eager (device-generic for CUDA coverage).
        m = torch.nn.Linear(4, 3).to(device).eval()
        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        mark_unbacked(x, 0, strict=True)
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)]
        )
        self.assertIn("USER_INPUT_SHAPES = [(None, 4)]", code)
        f_c = torch.compiler.precompile.load(code, cache)
        for bs in (8, 16, 2):
            xt = make_tensor((bs, 4), device=device, dtype=torch.float32)
            self.assertEqual(f_c(m, xt), m(xt))

    def test_unbacked_zero_batch_runs(self, device):
        # bs=0 on an unbacked dynamic dim is a valid runtime size (the symbol is >= 0);
        # the artifact runs on an empty batch and matches eager.
        m = torch.nn.Linear(4, 3).to(device).eval()
        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        mark_unbacked(x, 0)
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        xt = make_tensor((0, 4), device=device, dtype=torch.float32)
        self.assertEqual(f_c(m, xt), m(xt))

    def test_channels_last_marked_input_roundtrips(self, device):
        # A channels_last-marked dynamic input round-trips at the SAME layout for a
        # LAYOUT-PRESERVING (pointwise) op: _detect_memory_format records channels_last so
        # the refaked leaf preserves it, and the artifact accepts a channels_last runtime
        # input (matching eager). (conv output has a separate inductor layout limitation,
        # so this uses a pointwise op.)
        x = make_tensor((2, 3, 4, 4), device=device, dtype=torch.float32)
        x = x.to(memory_format=torch.channels_last)
        self.assertTrue(x.is_contiguous(memory_format=torch.channels_last))
        mark_unbacked(x, 0)
        code, cache = torch.compiler.precompile(
            lambda t: torch.relu(t) * 2.0, example_inputs=[(x,)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        xt = make_tensor((5, 3, 4, 4), device=device, dtype=torch.float32)
        xt = xt.to(memory_format=torch.channels_last)
        out = f_c(xt)
        self.assertEqual(out, torch.relu(xt) * 2.0)

    def test_marked_exotic_layout_rejected(self, device):
        # _detect_memory_format cannot preserve a layout that is neither contiguous nor
        # channels_last(_3d) through the refake, so a mark_unbacked input in such a layout
        # (here a transposed, non-contiguous 2D tensor) is rejected LOUDLY at capture rather
        # than silently forced contiguous (which would bake a wrong assert_size_stride).
        # Transpose makes a non-contiguous (8, 4) tensor in neither channels_last format.
        x = make_tensor((4, 8), device=device, dtype=torch.float32).t()
        self.assertFalse(x.is_contiguous())
        mark_unbacked(x, 0)
        with self.assertRaisesRegex(PrecompileError, "memory format"):
            torch.compiler.precompile(
                lambda t: t.contiguous() * 2.0, example_inputs=[(x,)]
            )

    def test_eager_backend_input_mutation(self, device):
        # The eager backend replays the raw ATen graph, so input mutation is reflected on
        # the passed tensor and matches eager, like the inductor backend.
        scratch = make_tensor((4,), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(
            lambda a: a.add_(1.0), example_inputs=[(scratch,)], backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        x = torch.zeros(4, device=device)
        out = f_c(x)
        self.assertEqual(x, torch.ones(4, device=device))
        self.assertEqual(out, torch.ones(4, device=device))

    def test_eager_backend_output_alias(self, device):
        # The eager backend reproduces an output that aliases an input (a view), matching
        # eager, via the raw ATen replay.
        x = make_tensor((2, 3), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(
            lambda a: a.t(), example_inputs=[(x,)], backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(x), x.t())


instantiate_device_type_tests(TestPrecompileNumerics, globals())


if __name__ == "__main__":
    run_tests()

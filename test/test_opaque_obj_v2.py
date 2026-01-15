# Owner(s): ["module: custom-operators"]

import gc
import random
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Optional

import torch
import torch.utils._pytree as pytree
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import AotEagerAndRecordGraphs, CompileCounter
from torch._dynamo.utils import counters as dynamo_counters
from torch._functorch.aot_autograd import (
    aot_compile_joint_with_descriptors,
    aot_export_joint_with_descriptors,
    aot_export_module,
)
from torch._library.effects import EffectType
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import (
    _OPAQUE_TYPES,
    get_opaque_type_name,
    is_opaque_type,
    is_opaque_value_type,
    register_opaque_type,
)
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.graph import _illegal_char_regex
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


class OpaqueQueue:
    def __init__(self, queue: list[torch.Tensor], init_tensor_: torch.Tensor) -> None:
        super().__init__()
        self.queue = queue
        self.init_tensor_ = init_tensor_

        # For testing purposes
        self._push_counter = 0
        self._pop_counter = 0
        self._size_counter = 0

    def push(self, tensor: torch.Tensor) -> None:
        self._push_counter += 1
        self.queue.append(tensor)

    def pop(self) -> torch.Tensor:
        self._pop_counter += 1
        if len(self.queue) > 0:
            return self.queue.pop(0)
        return self.init_tensor_

    def size(self) -> int:
        self._size_counter += 1
        return len(self.queue)


class RNGState:
    def __init__(self, seed):
        self.seed = seed
        self.rng = random.Random(self.seed)


class Counter:
    def __init__(self, start):
        self.counter = torch.tensor(start)

    def increment_counter(self):
        self.counter += 1


class AddModule(torch.nn.Module):
    def forward(self, x, y):
        return x * y


class ValueConfig:
    def __init__(self, mode: str):
        self.mode = mode

    def __eq__(self, other):
        return isinstance(other, ValueConfig) and self.mode == other.mode

    def __hash__(self):
        return hash(self.mode)

    def __fx_repr__(self):
        return f"ValueConfig(mode={self.mode!r})", {"ValueConfig": ValueConfig}


class SizeStore:
    def __init__(self, size: int):
        self.size = size

    def __eq__(self, other):
        return isinstance(other, ValueConfig) and self.size == other.size

    def __hash__(self):
        return hash(self.size)

    def __fx_repr__(self):
        # Return (repr_string, dict_mapping_name_to_type)
        return f"SizeStore(size={self.size!r})", {"SizeStore": SizeStore}

    def increment_size(self):
        return self.size + 1


class NestedValueSize:
    def __init__(self, size: SizeStore, config: ValueConfig):
        self.size = size
        self.config = config

    def __eq__(self, other):
        return self.size == other.size and self.config == other.config

    def __hash__(self):
        return hash(self.size) ^ hash(self.config)

    def __fx_repr__(self):
        # Recursively call __fx_repr__ on nested opaque objects
        size_eval, size_globals = self.size.__fx_repr__()
        config_eval, config_globals = self.config.__fx_repr__()

        # Combine repr and globals
        repr_str = f"NestedValueSize(size={size_eval}, config={config_eval})"
        all_globals = (
            {"NestedValueSize": NestedValueSize} | size_globals | config_globals
        )

        return repr_str, all_globals


register_opaque_type(OpaqueQueue, typ="reference")
register_opaque_type(RNGState, typ="reference")
register_opaque_type(Counter, typ="reference")
register_opaque_type(AddModule, typ="reference")
register_opaque_type(ValueConfig, typ="value")
register_opaque_type(SizeStore, typ="value")
register_opaque_type(NestedValueSize, typ="value")


class TestOpaqueObject(TestCase):
    def setUp(self):
        self.lib = torch.library.Library("_TestOpaqueObject", "FRAGMENT")  # noqa: TOR901

        torch.library.define(
            "_TestOpaqueObject::queue_push",
            f"({get_opaque_type_name(OpaqueQueue)} a, Tensor b) -> ()",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::queue_push", "CompositeExplicitAutograd", lib=self.lib
        )
        def push_impl(queue: OpaqueQueue, b: torch.Tensor) -> None:
            assert isinstance(queue, OpaqueQueue)
            queue.push(b)

        @torch.library.register_fake("_TestOpaqueObject::queue_push", lib=self.lib)
        def push_impl_fake(q: OpaqueQueue, b: torch.Tensor) -> None:
            pass

        self.lib.define(
            f"queue_pop({get_opaque_type_name(OpaqueQueue)} a) -> Tensor",
        )

        def pop_impl(queue: OpaqueQueue) -> torch.Tensor:
            assert isinstance(queue, OpaqueQueue)
            return queue.pop()

        self.lib.impl("queue_pop", pop_impl, "CompositeExplicitAutograd")

        def pop_impl_fake(q: OpaqueQueue) -> torch.Tensor:
            # This is not accurate since the queue could have tensors that are
            # not rank 1
            ctx = torch.library.get_ctx()
            u0 = ctx.new_dynamic_size()
            return torch.empty(u0)

        self.lib._register_fake("queue_pop", pop_impl_fake)

        @torch.library.custom_op(
            "_TestOpaqueObject::queue_size",
            mutates_args=[],
        )
        def size_impl(queue: OpaqueQueue) -> int:
            assert isinstance(queue, OpaqueQueue)
            return queue.size()

        @size_impl.register_fake
        def size_impl_fake(q: OpaqueQueue) -> int:
            ctx = torch._custom_op.impl.get_ctx()
            u0 = ctx.new_dynamic_size()
            torch._check(u0 >= 0)
            return u0

        torch.library.define(
            "_TestOpaqueObject::noisy_inject",
            f"(Tensor x, {get_opaque_type_name(RNGState)} obj) -> Tensor",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::noisy_inject", "CompositeExplicitAutograd", lib=self.lib
        )
        def noisy_inject(x: torch.Tensor, rng_state: RNGState) -> torch.Tensor:
            assert isinstance(rng_state, RNGState)
            out = x.clone()
            for i in range(out.numel()):
                out.view(-1)[i] += rng_state.rng.random()
            return out

        @torch.library.register_fake("_TestOpaqueObject::noisy_inject", lib=self.lib)
        def noisy_inject_fake(x: torch.Tensor, obj: RNGState) -> torch.Tensor:
            return torch.empty_like(x)

        @torch.library.custom_op(
            "_TestOpaqueObject::increment_counter",
            mutates_args=["prev"],
        )
        def increment_counter_impl(c: Counter, prev: torch.Tensor) -> torch.Tensor:
            assert isinstance(c, Counter)
            prev.copy_(c.counter)
            c.increment_counter()
            return c.counter

        @increment_counter_impl.register_fake
        def increment_counter_fake(c: Counter, prev: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(prev)

        torch.library.define(
            "_TestOpaqueObject::process_with_config",
            f"(Tensor x, {get_opaque_type_name(ValueConfig)} config) -> Tensor",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::process_with_config",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def process_with_config_impl(
            x: torch.Tensor, config: ValueConfig
        ) -> torch.Tensor:
            assert isinstance(config, ValueConfig)
            if config.mode == "square":
                return x * x
            elif config.mode == "double":
                return x + x
            else:
                return x.clone()

        @torch.library.register_fake(
            "_TestOpaqueObject::process_with_config", lib=self.lib
        )
        def process_with_config_fake(
            x: torch.Tensor, config: ValueConfig
        ) -> torch.Tensor:
            return torch.empty_like(x)

        torch.library.define(
            "_TestOpaqueObject::process_nested_config",
            f"(Tensor x, {get_opaque_type_name(NestedValueSize)} config) -> Tensor",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::process_nested_config",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def process_nested_config_impl(
            x: torch.Tensor, config: NestedValueSize
        ) -> torch.Tensor:
            assert isinstance(config, NestedValueSize)
            if config.config.mode == "square":
                return x * x
            elif config.config.mode == "double":
                return x + x
            else:
                return x.clone()

        @torch.library.register_fake(
            "_TestOpaqueObject::process_nested_config", lib=self.lib
        )
        def process_nested_config_fake(
            x: torch.Tensor, config: ValueConfig
        ) -> torch.Tensor:
            return torch.empty_like(x)

        torch.library.define(
            "_TestOpaqueObject::process_multiple_sizes",
            f"(Tensor x, {get_opaque_type_name(SizeStore)}[]? sizes) -> Tensor",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::process_multiple_sizes",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def process_multiple_sizes_impl(
            x: torch.Tensor, config: Optional[list[SizeStore]]
        ) -> torch.Tensor:
            if config is None:
                return x.clone()
            else:
                x_res = x.clone()
                for size in config:
                    x_res += size.size
                return x_res

        @torch.library.register_fake(
            "_TestOpaqueObject::process_multiple_sizes", lib=self.lib
        )
        def process_multiple_sizes_fake(
            x: torch.Tensor, config: Optional[list[SizeStore]]
        ) -> torch.Tensor:
            return torch.empty_like(x)

        super().setUp()

    def tearDown(self):
        self.lib._destroy()

        super().tearDown()

    def test_ops(self):
        queue = OpaqueQueue([], torch.zeros(3))

        torch.ops._TestOpaqueObject.queue_push(queue, torch.ones(3) + 1)
        size = torch.ops._TestOpaqueObject.queue_size(queue)
        self.assertEqual(size, 1)
        popped = torch.ops._TestOpaqueObject.queue_pop(queue)
        self.assertEqual(popped, torch.ones(3) + 1)
        size = torch.ops._TestOpaqueObject.queue_size(queue)
        self.assertEqual(size, 0)

    @parametrize("make_fx_tracing_mode", ["fake", "symbolic"])
    def test_make_fx(self, make_fx_tracing_mode):
        class M(torch.nn.Module):
            def forward(self, queue, x):
                torch.ops._TestOpaqueObject.queue_push(queue, x.tan())
                torch.ops._TestOpaqueObject.queue_push(queue, x.cos())
                torch.ops._TestOpaqueObject.queue_push(queue, x.sin())
                pop1 = torch.ops._TestOpaqueObject.queue_pop(queue)
                size1 = torch.ops._TestOpaqueObject.queue_size(queue)
                pop2 = torch.ops._TestOpaqueObject.queue_pop(queue)
                size2 = torch.ops._TestOpaqueObject.queue_size(queue)
                x_cos = pop1 + size1
                x_sin = pop2 - size2
                return x_sin + x_cos

        q1 = OpaqueQueue([], torch.empty(0).fill_(-1))
        q2 = OpaqueQueue([], torch.empty(0).fill_(-1))

        x = torch.ones(2, 3)
        gm = make_fx(M(), tracing_mode=make_fx_tracing_mode)(q1, x)
        self.assertTrue(torch.allclose(gm(q1, x), M()(q2, x)))
        self.assertEqual(q1._push_counter, 3)
        self.assertEqual(q1._pop_counter, 2)
        self.assertEqual(q1._size_counter, 2)
        self.assertEqual(q1.size(), 1)
        self.assertExpectedInline(
            gm.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1):
    tan = torch.ops.aten.tan.default(arg1_1)
    queue_push = torch.ops._TestOpaqueObject.queue_push.default(arg0_1, tan);  tan = queue_push = None
    cos = torch.ops.aten.cos.default(arg1_1)
    queue_push_1 = torch.ops._TestOpaqueObject.queue_push.default(arg0_1, cos);  cos = queue_push_1 = None
    sin = torch.ops.aten.sin.default(arg1_1);  arg1_1 = None
    queue_push_2 = torch.ops._TestOpaqueObject.queue_push.default(arg0_1, sin);  sin = queue_push_2 = None
    queue_pop = torch.ops._TestOpaqueObject.queue_pop.default(arg0_1)
    queue_size = torch.ops._TestOpaqueObject.queue_size.default(arg0_1)
    queue_pop_1 = torch.ops._TestOpaqueObject.queue_pop.default(arg0_1)
    queue_size_1 = torch.ops._TestOpaqueObject.queue_size.default(arg0_1);  arg0_1 = None
    add = torch.ops.aten.add.Tensor(queue_pop, queue_size);  queue_pop = queue_size = None
    sub = torch.ops.aten.sub.Tensor(queue_pop_1, queue_size_1);  queue_pop_1 = queue_size_1 = None
    add_1 = torch.ops.aten.add.Tensor(sub, add);  sub = add = None
    return add_1
    """,
        )

    @parametrize("make_fx_tracing_mode", ["fake", "symbolic"])
    def test_bad_fake(self, make_fx_tracing_mode):
        torch.library.define(
            "_TestOpaqueObject::bad_fake",
            f"(Tensor x, {get_opaque_type_name(RNGState)} obj) -> Tensor",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        def f(q, x):
            torch.ops._TestOpaqueObject.bad_fake(x, q)
            return x.cos()

        def bad_fake1(x, rng_state) -> torch.Tensor:
            self.assertTrue(isinstance(rng_state, FakeScriptObject))
            out = x.clone()
            for i in range(out.numel()):
                out.view(-1)[i] += rng_state.rng.random()  # bad: accessing attributes
            return out

        torch.library.register_fake(
            "_TestOpaqueObject::bad_fake", bad_fake1, lib=self.lib, allow_override=True
        )

        with self.assertRaisesRegex(
            AttributeError,
            "Tried to call __getattr__ with attr",
        ):
            make_fx(f, tracing_mode=make_fx_tracing_mode)(RNGState(0), torch.ones(3))

        def bad_fake2(x, rng_state) -> torch.Tensor:
            rng_state.rng = "foo"
            return torch.empty_like(x)

        torch.library.register_fake(
            "_TestOpaqueObject::bad_fake", bad_fake2, lib=self.lib, allow_override=True
        )

        with self.assertRaisesRegex(
            AttributeError,
            "Tried to call __setattr__ with attr",
        ):
            make_fx(f, tracing_mode=make_fx_tracing_mode)(RNGState(0), torch.ones(3))

    def test_aot_export(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, rng_state, x):
                x = torch.ops._TestOpaqueObject.noisy_inject(x, rng_state)
                x = x * x
                x = torch.ops._TestOpaqueObject.noisy_inject(x, rng_state)
                x = x + x
                return (x,)

        mod = Model()
        rng = RNGState(0)
        x = torch.ones(2, 3)

        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
        fake_rng = torch._library.fake_class_registry.maybe_to_fake_obj(fake_mode, rng)
        fake_x = fake_mode.from_tensor(x)
        gm = aot_export_module(mod, (fake_rng, fake_x), trace_joint=False)[0]

        # By default we don't register ops containing PyObjs as being effectful
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    noisy_inject = torch.ops._TestOpaqueObject.noisy_inject.default(arg1_1, arg0_1);  arg1_1 = None
    mul = torch.ops.aten.mul.Tensor(noisy_inject, noisy_inject);  noisy_inject = None
    noisy_inject_1 = torch.ops._TestOpaqueObject.noisy_inject.default(mul, arg0_1);  mul = arg0_1 = None
    add = torch.ops.aten.add.Tensor(noisy_inject_1, noisy_inject_1);  noisy_inject_1 = None
    return (add,)""",  # noqa: B950
        )

        torch.library._register_effectful_op(
            "_TestOpaqueObject::noisy_inject", EffectType.ORDERED
        )
        try:
            gm = aot_export_module(mod, (rng, fake_x), trace_joint=False)[0]
            # inputs: token, rng, x
            # return: token, res
            self.assertExpectedInline(
                gm.code.strip(),
                """\
def forward(self, arg0_1, arg1_1, arg2_1):
    with_effects = torch.ops.higher_order.with_effects(arg0_1, torch.ops._TestOpaqueObject.noisy_inject.default, arg2_1, arg1_1);  arg0_1 = arg2_1 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    mul = torch.ops.aten.mul.Tensor(getitem_1, getitem_1);  getitem_1 = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops._TestOpaqueObject.noisy_inject.default, mul, arg1_1);  getitem = mul = arg1_1 = None
    getitem_2 = with_effects_1[0]
    getitem_3 = with_effects_1[1];  with_effects_1 = None
    add = torch.ops.aten.add.Tensor(getitem_3, getitem_3);  getitem_3 = None
    return (getitem_2, add)""",  # noqa: B950
            )
        finally:
            torch.library._register_effectful_op(
                "_TestOpaqueObject::noisy_inject", None
            )

    def test_compile1(self):
        def foo(rng_state, x):
            x = torch.ops._TestOpaqueObject.noisy_inject(x, rng_state)
            x = x * x
            x = torch.ops._TestOpaqueObject.noisy_inject(x, rng_state)
            x = x + x
            return x

        rng = RNGState(0)
        x = torch.ones(2, 3)

        res = torch.compile(foo, fullgraph=True, backend="inductor")(rng, x)
        self.assertFalse(torch.allclose(res, x * x + x))

        backend = AotEagerAndRecordGraphs()
        torch.compile(foo, fullgraph=True, backend=backend)(rng, x)

        # This is done in torch.fx's graph in _namespace.create_name() where it
        # sanitizes the name
        fx_class = _illegal_char_regex.sub("_", get_opaque_type_name(RNGState))
        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            f"""\
def forward(self, L_x_ : torch.Tensor, L_rng_state_ : {fx_class}):
    l_x_ = L_x_
    l_rng_state_ = L_rng_state_
    x = torch.ops._TestOpaqueObject.noisy_inject(l_x_, l_rng_state_);  l_x_ = None
    x_1 = x * x;  x = None
    x_2 = torch.ops._TestOpaqueObject.noisy_inject(x_1, l_rng_state_);  x_1 = l_rng_state_ = None
    x_3 = x_2 + x_2;  x_2 = None
    return (x_3,)""",  # noqa: B950
        )
        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    noisy_inject = torch.ops._TestOpaqueObject.noisy_inject.default(arg0_1, arg1_1);  arg0_1 = None
    mul = torch.ops.aten.mul.Tensor(noisy_inject, noisy_inject);  noisy_inject = None
    noisy_inject_1 = torch.ops._TestOpaqueObject.noisy_inject.default(mul, arg1_1);  mul = arg1_1 = None
    add = torch.ops.aten.add.Tensor(noisy_inject_1, noisy_inject_1);  noisy_inject_1 = None
    return (add,)""",  # noqa: B950
        )

    def test_compile_global(self):
        counter = Counter(0)

        def foo(x, y):
            z = torch.ops._TestOpaqueObject.increment_counter(counter, y)
            x = x * z
            z = torch.ops._TestOpaqueObject.increment_counter(counter, y)
            x = x + z
            return x, counter

        inp = (torch.tensor(1), torch.tensor(0))
        backend = AotEagerAndRecordGraphs()
        opt_f = torch.compile(foo, fullgraph=True, backend=backend)
        res = opt_f(*inp)
        self.assertEqual(res[0], torch.tensor(3))
        self.assertEqual(res[1].counter, torch.tensor(2))

        res = opt_f(*inp)
        self.assertEqual(res[0], torch.tensor(7))
        self.assertEqual(res[1].counter, torch.tensor(4))

        # counter is automatically lifted as an input
        # Even though we returned counter in the eager code, it does not get
        # returned in the graph because dynamo does not detect that the object
        # is mutated.
        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops._TestOpaqueObject.increment_counter.default, c = arg1_1, _prev_base_index = 0, _all_bases = [arg0_1])
    getitem = auto_functionalized_v2[0]
    getitem_1 = auto_functionalized_v2[1];  auto_functionalized_v2 = None
    mul = torch.ops.aten.mul.Tensor(arg2_1, getitem);  arg2_1 = getitem = None
    auto_functionalized_v2_1 = torch.ops.higher_order.auto_functionalized_v2(torch.ops._TestOpaqueObject.increment_counter.default, c = arg1_1, _prev_base_index = 0, _all_bases = [getitem_1]);  arg1_1 = getitem_1 = None
    getitem_2 = auto_functionalized_v2_1[0]
    getitem_3 = auto_functionalized_v2_1[1];  auto_functionalized_v2_1 = None
    add = torch.ops.aten.add.Tensor(mul, getitem_2);  mul = getitem_2 = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, getitem_3);  arg0_1 = getitem_3 = copy_ = None
    return (add,)""",  # noqa: B950
        )

    def test_compile_create_intermediate(self):
        dynamo_counters.clear()

        def foo(x, y):
            counter = Counter(0)
            z = torch.ops._TestOpaqueObject.increment_counter(counter, y)
            x = x * z
            return x

        inp = (torch.tensor(1), torch.tensor(0))
        torch.compile(foo)(*inp)
        self.assertEqual(len(dynamo_counters["graph_break"]), 1)
        self.assertTrue(
            "An opaque object was created in the middle of the program"
            in next(iter(dynamo_counters["graph_break"].keys())),
        )

    def test_compile_attribute(self):
        counter = Counter(0)

        def foo(counter, x):
            x = x * x
            counter.increment_counter()
            return x

        with self.assertRaisesRegex(
            RuntimeError, "Attempted to access attributes/methods on an OpaqueObject"
        ):
            torch.compile(foo)(counter, torch.ones(2, 3))

        def bar(counter, x):
            x = x * x
            x += counter.counter
            return x

        with self.assertRaisesRegex(
            RuntimeError, "Attempted to access attributes/methods on an OpaqueObject"
        ):
            torch.compile(bar)(counter, torch.ones(2, 3))

    def test_export_joint(self):
        torch.library.define(
            "_TestOpaqueObject::module_mul",
            f"({get_opaque_type_name(AddModule)} a, Tensor b, SymInt c) -> Tensor",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::module_mul", "CompositeExplicitAutograd", lib=self.lib
        )
        def module_mul_impl(m: AddModule, a: torch.Tensor, b: int) -> torch.Tensor:
            assert isinstance(m, AddModule)
            return m(a, b)

        @torch.library.register_fake("_TestOpaqueObject::module_mul", lib=self.lib)
        def module_mul_fake(m: AddModule, a: torch.Tensor, b: int) -> torch.Tensor:
            return torch.empty_like(a)

        def module_mul_setup_context(ctx, inputs, output):
            m, a, b = inputs
            ctx.b = b

        def module_mul_backward(ctx, grad) -> torch.Tensor:
            return None, grad * ctx.b, None

        torch.library.register_autograd(
            "_TestOpaqueObject::module_mul",
            module_mul_backward,
            setup_context=module_mul_setup_context,
            lib=self.lib,
        )

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.moo = AddModule()

            def forward(self, x, y):
                b = y.item()
                return torch.ops._TestOpaqueObject.module_mul(self.moo, x, b)

        inp = (torch.randn(3, requires_grad=True), torch.tensor(4))
        with ExitStack() as stack:
            with FakeTensorMode(shape_env=ShapeEnv()):
                joint = aot_export_joint_with_descriptors(stack, M(), inp)
                self.assertExpectedInline(
                    joint.graph_module.code.strip(),
                    """\
def forward(self, primals, tangents):
    primals_1, primals_2, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(primals_2);  primals_2 = None
    _opaque_obj0 = self._opaque_obj0
    module_mul = torch.ops._TestOpaqueObject.module_mul.default(_opaque_obj0, primals_1, _local_scalar_dense);  _opaque_obj0 = primals_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(tangents_1, _local_scalar_dense);  tangents_1 = _local_scalar_dense = None
    return pytree.tree_unflatten([module_mul, mul_1, None], self._out_spec)""",  # noqa: B950
                )
                compiled_fn = aot_compile_joint_with_descriptors(joint)

        self.assertEqual(compiled_fn(*inp), M()(*inp))

    def test_invalid_value_type(self):
        class NoEq:
            def __init__(self, x):
                self.x = x

        with self.assertRaisesRegex(
            TypeError, "expected to have a non-default `__eq__`"
        ):
            register_opaque_type(NoEq, typ="value")

        class NoHash:
            def __init__(self, x):
                self.x = x

            def __eq__(self, other):
                return self.x == other.x

        with self.assertRaisesRegex(
            TypeError, "expected to have a non-default `__hash__`"
        ):
            register_opaque_type(NoHash, typ="value")

        class NoRepr:
            def __init__(self, x):
                self.x = x

            def __eq__(self, other):
                return self.x == other.x

            def __hash__(self):
                return hash(self.x)

        with self.assertRaisesRegex(TypeError, "expected to have a `__fx_repr__`"):
            register_opaque_type(NoRepr, typ="value")

    def test_invalid_schema(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "unknown type specifier",
        ):
            torch.library.define(
                "_TestOpaqueObject::invalid_op1",
                "(foo.bar.baz a) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=self.lib,
            )

        with self.assertRaisesRegex(
            RuntimeError,
            r"expected \) but found 'dots' here",
        ):
            torch.library.define(
                "_TestOpaqueObject::invalid_op2",
                "(......... a) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=self.lib,
            )

        with self.assertRaisesRegex(
            RuntimeError,
            "unknown type specifier",
        ):
            torch.library.define(
                "_TestOpaqueObject::invalid_op5",
                "(MyNamespace..MyClass a) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=self.lib,
            )

    def test_invalid_opaque_obj_types(self):
        for t in [str, bool, int, float, torch.Tensor]:
            with self.assertRaisesRegex(ValueError, "Unable to register built-in type"):
                register_opaque_type(t, typ="reference")

        @dataclass
        class Bad1:
            x: int

        pytree.register_dataclass(Bad1)
        try:
            with self.assertRaisesRegex(
                ValueError,
                "cannot be registered as an opaque object as it has been registered as a pytree.",
            ):
                register_opaque_type(Bad1, typ="reference")
        finally:
            # Clean up pytree registration to avoid leaking state to other tests
            pytree.SUPPORTED_NODES.pop(Bad1, None)
            pytree.SUPPORTED_SERIALIZED_TYPES.pop(Bad1, None)
            pytree.CONSTANT_NODES.discard(Bad1)

        @dataclass
        class Bad2:
            x: int

        register_opaque_type(Bad2, typ="reference")
        with self.assertRaisesRegex(
            ValueError,
            "cannot be registered as a pytree as it has been registered as an opaque object.",
        ):
            pytree.register_dataclass(Bad2)

    def test_value_type_recompile(self):
        cnt = CompileCounter()

        def foo(x, cfg):
            return torch.ops._TestOpaqueObject.process_with_config(x, cfg)

        x = torch.randn(3, 3)

        opt_f = torch.compile(foo, backend=cnt, fullgraph=True)
        res = opt_f(x, ValueConfig("square"))
        self.assertEqual(res, x * x)
        self.assertEqual(cnt.frame_count, 1)

        res = opt_f(x, ValueConfig("square"))
        self.assertEqual(res, x * x)
        self.assertEqual(cnt.frame_count, 1)

        # Recompilation!
        res = opt_f(x, ValueConfig("double"))
        self.assertEqual(res, x + x)
        self.assertEqual(cnt.frame_count, 2)

    def test_value_type_graph_input(self):
        # Even though cfg is an input, it should not be an input to the dynamo
        # graph. Instead it should directly put in the graph argument as a
        # constant.

        def foo(x, cfg):
            return torch.ops._TestOpaqueObject.process_with_config(x, cfg)

        x = torch.randn(3, 3)
        backend = AotEagerAndRecordGraphs()
        opt_f = torch.compile(foo, fullgraph=True, backend=backend)
        opt_f(x, ValueConfig("square"))

        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    process_with_config = torch.ops._TestOpaqueObject.process_with_config(l_x_, ValueConfig(mode='square'));  l_x_ = None
    return (process_with_config,)""",  # noqa: B950
        )
        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1):
    process_with_config = torch.ops._TestOpaqueObject.process_with_config.default(arg0_1, ValueConfig(mode='square'));  arg0_1 = None
    return (process_with_config,)""",  # noqa: B950
        )

        opt_f(x, ValueConfig("double"))

        self.assertExpectedInline(
            backend.fw_graphs[1].code.strip(),
            """\
def forward(self, arg0_1):
    process_with_config = torch.ops._TestOpaqueObject.process_with_config.default(arg0_1, ValueConfig(mode='double'));  arg0_1 = None
    return (process_with_config,)""",  # noqa: B950
        )

    def test_value_type_graph_intermediate(self):
        def foo(x, config):
            cfg = ValueConfig(config)
            return torch.ops._TestOpaqueObject.process_with_config(x, cfg)

        x = torch.randn(3, 3)
        backend = AotEagerAndRecordGraphs()
        opt_f = torch.compile(foo, fullgraph=True, backend=backend)
        opt_f(x, "square")

        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    process_with_config = torch.ops._TestOpaqueObject.process_with_config(l_x_, ValueConfig(mode='square'));  l_x_ = None
    return (process_with_config,)""",  # noqa: B950
        )
        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1):
    process_with_config = torch.ops._TestOpaqueObject.process_with_config.default(arg0_1, ValueConfig(mode='square'));  arg0_1 = None
    return (process_with_config,)""",  # noqa: B950
        )

        opt_f(x, "double")
        self.assertExpectedInline(
            backend.fw_graphs[1].code.strip(),
            """\
def forward(self, arg0_1):
    process_with_config = torch.ops._TestOpaqueObject.process_with_config.default(arg0_1, ValueConfig(mode='double'));  arg0_1 = None
    return (process_with_config,)""",  # noqa: B950
        )

        opt_f = torch.compile(foo, fullgraph=True, backend="inductor")
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x, "square"), foo(x, "square"))
        self.assertEqual(opt_f(x, "double"), foo(x, "double"))

    def test_value_type_attr_access(self):
        def foo(x):
            size = SizeStore(x.shape[0])
            t1 = torch.ones((size.size,))
            t2 = torch.cat([x, t1])
            size.increment_size()
            return t2 + size.size

        x = torch.randn(3)
        backend = AotEagerAndRecordGraphs()
        opt_f = torch.compile(foo, fullgraph=True, backend=backend)
        res = opt_f(x)
        self.assertEqual(res, foo(x))

        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([3], device = device(type='cpu'), pin_memory = False)
    cat = torch.ops.aten.cat.default([arg0_1, ones]);  arg0_1 = ones = None
    add = torch.ops.aten.add.Tensor(cat, 3);  cat = None
    return (add,)""",  # noqa: B950
        )

    def test_weakref_cleanup(self):
        def register_tmp_class():
            class TmpClass:
                def __init__(self, value):
                    self.value = value

                def __eq__(self, other):
                    return self.value == other.value

                def __hash__(self):
                    return hash(self.value)

                def __fx_repr__(self):
                    return f"TmpClass(value={self.value!r})", {"TmpClass": TmpClass}

            register_opaque_type(TmpClass, typ="value")

            self.assertTrue(is_opaque_type(TmpClass))
            self.assertTrue(is_opaque_value_type(TmpClass))
            self.assertIn(TmpClass, _OPAQUE_TYPES)

            return get_opaque_type_name(TmpClass)

        # registers TmpClass as opaque
        tmp_class_name = register_tmp_class()

        # garbage collect TmpClass
        gc.collect()

        # Verify that the class is no longer registered
        for opaque_info in _OPAQUE_TYPES.values():
            self.assertFalse(tmp_class_name in opaque_info.class_name)

    def test_value_type_nested(self):
        def foo(x, config):
            size = SizeStore(x.shape[0])
            cfg = NestedValueSize(size, ValueConfig(config))
            return torch.ops._TestOpaqueObject.process_nested_config(x, cfg)

        x = torch.randn(3, 3)
        backend = AotEagerAndRecordGraphs()
        opt_f = torch.compile(foo, fullgraph=True, backend=backend)
        opt_f(x, "square")

        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1):
    process_nested_config = torch.ops._TestOpaqueObject.process_nested_config.default(arg0_1, NestedValueSize(size=SizeStore(size=3), config=ValueConfig(mode='square')));  arg0_1 = None
    return (process_nested_config,)""",  # noqa: B950
        )

        opt_f = torch.compile(foo, fullgraph=True, backend="inductor")
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x, "square"), foo(x, "square"))
        self.assertEqual(opt_f(x, "double"), foo(x, "double"))

    def test_value_type_list(self):
        def foo(x):
            sizes = []
            for size in x.size():
                sizes.append(SizeStore(size))
            return torch.ops._TestOpaqueObject.process_multiple_sizes(x, sizes)

        x = torch.randn(3, 3)
        backend = AotEagerAndRecordGraphs()
        opt_f = torch.compile(foo, fullgraph=True, backend=backend)
        opt_f(x)

        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1):
    process_multiple_sizes = torch.ops._TestOpaqueObject.process_multiple_sizes.default(arg0_1, [SizeStore(size=3), SizeStore(size=3)]);  arg0_1 = None
    return (process_multiple_sizes,)""",  # noqa: B950
        )

        opt_f = torch.compile(foo, fullgraph=True, backend="inductor")
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x), foo(x))


instantiate_parametrized_tests(TestOpaqueObject)


if __name__ == "__main__":
    run_tests()

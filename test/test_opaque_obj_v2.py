# Owner(s): ["module: custom-operators"]

import random
from contextlib import ExitStack

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import AotEagerAndRecordGraphs
from torch._dynamo.utils import counters as dynamo_counters
from torch._functorch.aot_autograd import (
    aot_compile_joint_with_descriptors,
    aot_export_joint_with_descriptors,
    aot_export_module,
)
from torch._library.effects import EffectType
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import get_opaque_type_name, register_opaque_type
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
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


class Moodule(torch.nn.Module):
    def forward(self, x, y):
        return x * y


register_opaque_type(OpaqueQueue)
register_opaque_type(RNGState)
register_opaque_type(Counter)
register_opaque_type(Moodule)


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
            torch._check_is_size(u0)
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

    def test_compile(self):
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
        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            f"""\
def forward(self, L_x_ : torch.Tensor, L_rng_state_ : {get_opaque_type_name(RNGState)}):
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
        self.assertExpectedInline(
            next(iter(dynamo_counters["graph_break"].keys())),
            """\
Opaque object were created in the middle of the program and passed to a custom op.
  Explanation: Opaque objects cannot be created inside the torch.compile region. They must be created before entering the compiled function.
  Hint: Please create the opaque object before calling torch.compile and pass it in as an argument or as a global variable.

  Developer debug context: Opaque object types: [<class '__main__.Counter'>]. Function: _TestOpaqueObject.increment_counter

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0363.html""",  # noqa: B950
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
            f"({get_opaque_type_name(Moodule)} a, Tensor b, SymInt c) -> Tensor",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::module_mul", "CompositeExplicitAutograd", lib=self.lib
        )
        def module_mul_impl(m: Moodule, a: torch.Tensor, b: int) -> torch.Tensor:
            assert isinstance(m, Moodule)
            return m(a, b)

        @torch.library.register_fake("_TestOpaqueObject::module_mul", lib=self.lib)
        def module_mul_fake(m: Moodule, a: torch.Tensor, b: int) -> torch.Tensor:
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
                self.moo = Moodule()

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


instantiate_parametrized_tests(TestOpaqueObject)


if __name__ == "__main__":
    run_tests()

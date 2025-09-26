# Owner(s): ["module: custom-operators"]
import copy
import random

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._functorch.aot_autograd import aot_export_module
from torch._higher_order_ops.effects import _deregister_effectful_op
from torch._library.opaque_object import get_payload, make_opaque, set_payload
from torch.fx.experimental.proxy_tensor import make_fx
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

    def __eq__(self, other):
        if len(self.queue) != len(other.queue):
            return False
        for q1, q2 in zip(self.queue, other.queue):
            if not torch.allclose(q1, q2):
                return False
        return torch.allclose(self.init_tensor_, other.init_tensor_)


class RNGState:
    def __init__(self, seed):
        self.rng = random.Random(seed)


class TestOpaqueObject(TestCase):
    def setUp(self):
        self.lib = torch.library.Library("_TestOpaqueObject", "FRAGMENT")  # noqa: TOR901

        torch.library.define(
            "_TestOpaqueObject::queue_push",
            "(__torch__.torch.classes.aten.OpaqueObject a, Tensor b) -> ()",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::queue_push", "CompositeExplicitAutograd", lib=self.lib
        )
        def push_impl(q: torch._C.ScriptObject, b: torch.Tensor) -> None:
            queue = get_payload(q)
            assert isinstance(queue, OpaqueQueue)
            queue.push(b)

        @torch.library.register_fake("_TestOpaqueObject::queue_push", lib=self.lib)
        def push_impl_fake(q: torch._C.ScriptObject, b: torch.Tensor) -> None:
            pass

        self.lib.define(
            "queue_pop(__torch__.torch.classes.aten.OpaqueObject a) -> Tensor",
        )

        def pop_impl(q: torch._C.ScriptObject) -> torch.Tensor:
            queue = get_payload(q)
            assert isinstance(queue, OpaqueQueue)
            return queue.pop()

        self.lib.impl("queue_pop", pop_impl, "CompositeExplicitAutograd")

        def pop_impl_fake(q: torch._C.ScriptObject) -> torch.Tensor:
            # This is not accurate
            ctx = torch._custom_op.impl.get_ctx()
            u0 = ctx.new_dynamic_size()
            return torch.empty(u0)

        self.lib._register_fake("queue_pop", pop_impl_fake)

        @torch.library.custom_op(
            "_TestOpaqueObject::queue_size",
            mutates_args=[],
        )
        def size_impl(q: torch.library.OpaqueType) -> int:
            queue = get_payload(q)
            assert isinstance(queue, OpaqueQueue)
            return queue.size()

        @size_impl.register_fake
        def size_impl_fake(q: torch._C.ScriptObject) -> int:
            ctx = torch._custom_op.impl.get_ctx()
            u0 = ctx.new_dynamic_size()
            return u0

        torch.library.define(
            "_TestOpaqueObject::noisy_inject",
            "(Tensor x, __torch__.torch.classes.aten.OpaqueObject obj) -> Tensor",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::noisy_inject", "CompositeExplicitAutograd", lib=self.lib
        )
        def noisy_inject(x: torch.Tensor, obj: torch._C.ScriptObject) -> torch.Tensor:
            rng_state = get_payload(obj)
            assert isinstance(rng_state, RNGState)
            out = x.clone()
            for i in range(out.numel()):
                out.view(-1)[i] += rng_state.rng.random()
            return out

        @torch.library.register_fake("_TestOpaqueObject::noisy_inject", lib=self.lib)
        def noisy_inject_fake(
            x: torch.Tensor, obj: torch._C.ScriptObject
        ) -> torch.Tensor:
            return torch.empty_like(x)

        super().setUp()

    def tearDown(self):
        self.lib._destroy()

        super().tearDown()

    def test_creation(self):
        queue = OpaqueQueue([], torch.zeros(3))
        obj = make_opaque(queue)
        self.assertTrue(isinstance(obj, torch._C.ScriptObject))
        self.assertEqual(str(obj._type()), "__torch__.torch.classes.aten.OpaqueObject")

        # obj.payload stores a direct reference to this python queue object
        payload = get_payload(obj)
        self.assertEqual(payload, queue)
        queue.push(torch.ones(3))
        self.assertEqual(payload.size(), 1)

    def test_ops(self):
        queue = OpaqueQueue([], torch.zeros(3))
        obj = make_opaque()
        set_payload(obj, queue)

        torch.ops._TestOpaqueObject.queue_push(obj, torch.ones(3) + 1)
        self.assertEqual(queue.size(), 1)
        size = torch.ops._TestOpaqueObject.queue_size(obj)
        self.assertEqual(size, queue.size())
        popped = torch.ops._TestOpaqueObject.queue_pop(obj)
        self.assertEqual(popped, torch.ones(3) + 1)
        self.assertEqual(queue.size(), 0)

    def test_eq(self):
        self.assertTrue(make_opaque("moo") == make_opaque("moo"))
        self.assertFalse(make_opaque("moo") == make_opaque("mop"))

        q1 = OpaqueQueue([torch.ones(3)], torch.zeros(3))
        q2 = OpaqueQueue([torch.ones(3)], torch.zeros(3))
        obj1 = make_opaque(q1)
        obj2 = make_opaque(q2)
        self.assertTrue(obj1 == obj1)
        self.assertTrue(q1 == q2)
        self.assertTrue(obj1 == obj2)

    def test_deepcopy(self):
        q1 = OpaqueQueue([torch.ones(3), torch.ones(3) * 2], torch.zeros(3))
        obj1 = make_opaque(q1)

        obj2 = copy.deepcopy(obj1)
        q2 = get_payload(obj2)

        self.assertTrue(q1 is not q2)
        self.assertTrue(q1 == q2)

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
        obj1 = make_opaque(q1)
        q2 = OpaqueQueue([], torch.empty(0).fill_(-1))
        obj2 = make_opaque(q2)

        x = torch.ones(2, 3)
        gm = make_fx(M(), tracing_mode=make_fx_tracing_mode)(obj1, x)
        self.assertTrue(torch.allclose(gm(obj1, x), M()(obj2, x)))
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
        obj1 = make_opaque(rng)
        x = torch.ones(2, 3)

        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
        fake_tq1 = torch._library.fake_class_registry.maybe_to_fake_obj(fake_mode, obj1)
        fake_x = fake_mode.from_tensor(x)
        gm = aot_export_module(mod, (fake_tq1, fake_x), trace_joint=False)[0]

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

        # By default, ops with ScriptObjects as inputs are registered as being
        # effectful
        _deregister_effectful_op("_TestOpaqueObject::noisy_inject.default")

        # If we register with None, this means the ops do not have effect
        torch.library.register_effectful_op(
            "_TestOpaqueObject::noisy_inject.default", None
        )
        gm = aot_export_module(mod, (obj1, fake_x), trace_joint=False)[0]

        # There is no longer a token input, and no longer with_effect HOO
        # because the ops are marked as not effectful
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
        _deregister_effectful_op("_TestOpaqueObject::noisy_inject.default")


instantiate_parametrized_tests(TestOpaqueObject)

if __name__ == "__main__":
    run_tests()

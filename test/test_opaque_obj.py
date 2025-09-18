# Owner(s): ["module: custom-operators"]
import copy

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
            queue = get_payload(q)
            assert isinstance(queue, OpaqueQueue)
            queue.push(b)

        self.lib.define(
            "queue_pop(__torch__.torch.classes.aten.OpaqueObject a) -> Tensor",
        )

        def pop_impl(q: torch._C.ScriptObject) -> torch.Tensor:
            queue = get_payload(q)
            assert isinstance(queue, OpaqueQueue)
            return queue.pop()

        self.lib.impl("queue_pop", pop_impl, "CompositeExplicitAutograd")
        self.lib._register_fake("queue_pop", pop_impl)

        @torch.library.custom_op(
            "_TestOpaqueObject::queue_size",
            mutates_args=[],
            schema="(__torch__.torch.classes.aten.OpaqueObject a) -> int",
        )
        def size_impl(q: torch._C.ScriptObject) -> int:
            queue = get_payload(q)
            assert isinstance(queue, OpaqueQueue)
            return queue.size()

        @size_impl.register_fake
        def size_impl_fake(q: torch._C.ScriptObject) -> int:
            queue = get_payload(q)
            assert isinstance(queue, OpaqueQueue)
            return queue.size()

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

    @parametrize("make_fx_tracing_mode", ["fake", "symbolic"])
    def test_make_fx(self, make_fx_tracing_mode):
        class M(torch.nn.Module):
            def forward(self, queue, x):
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
        obj2 = make_opaque(q1)

        x = torch.ones(2, 3)
        gm = make_fx(M(), tracing_mode=make_fx_tracing_mode)(obj1, x)
        self.assertEqual(q1._push_counter, 2)
        self.assertEqual(q1._pop_counter, 2)
        self.assertEqual(q1._size_counter, 2)
        self.assertEqual(q1.size(), 0)
        self.assertExpectedInline(
            gm.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1):
    cos = torch.ops.aten.cos.default(arg1_1)
    queue_push = torch.ops._TestOpaqueObject.queue_push.default(arg0_1, cos);  cos = queue_push = None
    sin = torch.ops.aten.sin.default(arg1_1);  arg1_1 = None
    queue_push_1 = torch.ops._TestOpaqueObject.queue_push.default(arg0_1, sin);  sin = queue_push_1 = None
    queue_pop = torch.ops._TestOpaqueObject.queue_pop.default(arg0_1)
    queue_size = torch.ops._TestOpaqueObject.queue_size.default(arg0_1);  queue_size = None
    queue_pop_1 = torch.ops._TestOpaqueObject.queue_pop.default(arg0_1)
    queue_size_1 = torch.ops._TestOpaqueObject.queue_size.default(arg0_1);  arg0_1 = queue_size_1 = None
    add = torch.ops.aten.add.Tensor(queue_pop, 1);  queue_pop = None
    sub = torch.ops.aten.sub.Tensor(queue_pop_1, 0);  queue_pop_1 = None
    add_1 = torch.ops.aten.add.Tensor(sub, add);  sub = add = None
    return add_1
    """,
        )

        self.assertTrue(torch.allclose(gm(obj1, x), M()(obj2, x)))

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

    def test_aot_export_tensor_queue_operators(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, tq, x):
                torch.ops._TestOpaqueObject.queue_push(tq, x.cos())
                torch.ops._TestOpaqueObject.queue_push(tq, x.sin())
                x_sin = torch.ops._TestOpaqueObject.queue_pop(
                    tq
                ) - torch.ops._TestOpaqueObject.queue_size(tq)
                x_cos = torch.ops._TestOpaqueObject.queue_pop(
                    tq
                ) + torch.ops._TestOpaqueObject.queue_size(tq)
                return x_sin, x_cos, tq

        mod = Model()

        tq1 = OpaqueQueue([], torch.empty(0).fill_(-1))
        obj1 = make_opaque(tq1)
        x = torch.ones(2, 3)

        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
        # fake_tq1 = torch._library.fake_class_registry.maybe_to_fake_obj(fake_mode, tq1)
        fake_x = fake_mode.from_tensor(x)
        gm = aot_export_module(mod, (obj1, fake_x), trace_joint=False)[0]

        # inputs: token, tq, x
        # return: token, x_sin, x_cos, tq
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    cos = torch.ops.aten.cos.default(arg2_1)
    with_effects = torch.ops.higher_order.with_effects(arg0_1, torch.ops._TestOpaqueObject.queue_push.default, arg1_1, cos);  arg0_1 = cos = None
    getitem = with_effects[0];  with_effects = None
    sin = torch.ops.aten.sin.default(arg2_1);  arg2_1 = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops._TestOpaqueObject.queue_push.default, arg1_1, sin);  getitem = sin = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    with_effects_2 = torch.ops.higher_order.with_effects(getitem_2, torch.ops._TestOpaqueObject.queue_pop.default, arg1_1);  getitem_2 = None
    getitem_4 = with_effects_2[0]
    getitem_5 = with_effects_2[1];  with_effects_2 = None
    with_effects_3 = torch.ops.higher_order.with_effects(getitem_4, torch.ops._TestOpaqueObject.queue_size.default, arg1_1);  getitem_4 = None
    getitem_6 = with_effects_3[0];  with_effects_3 = None
    sub = torch.ops.aten.sub.Tensor(getitem_5, 1);  getitem_5 = None
    with_effects_4 = torch.ops.higher_order.with_effects(getitem_6, torch.ops._TestOpaqueObject.queue_pop.default, arg1_1);  getitem_6 = None
    getitem_8 = with_effects_4[0]
    getitem_9 = with_effects_4[1];  with_effects_4 = None
    with_effects_5 = torch.ops.higher_order.with_effects(getitem_8, torch.ops._TestOpaqueObject.queue_size.default, arg1_1);  getitem_8 = None
    getitem_10 = with_effects_5[0];  with_effects_5 = None
    add = torch.ops.aten.add.Tensor(getitem_9, 0);  getitem_9 = None
    return (getitem_10, sub, add, arg1_1)""",  # noqa: B950
        )

        # If we register with None, this should mean the ops do not have effect
        # since by default ops with ScriptObjects have effect
        torch.library.register_effectful_op(
            "_TestOpaqueObject::queue_push.default", None
        )
        torch.library.register_effectful_op(
            "_TestOpaqueObject::queue_pop.default", None
        )
        torch.library.register_effectful_op(
            "_TestOpaqueObject::queue_size.default", None
        )
        gm = aot_export_module(mod, (obj1, fake_x), trace_joint=False)[0]

        # inputs: token, tq, x
        # return: token, x_sin, x_cos, tq
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    cos = torch.ops.aten.cos.default(arg2_1)
    with_effects = torch.ops.higher_order.with_effects(arg0_1, torch.ops._TestOpaqueObject.queue_push.default, arg1_1, cos);  arg0_1 = cos = None
    getitem = with_effects[0];  with_effects = None
    sin = torch.ops.aten.sin.default(arg2_1);  arg2_1 = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops._TestOpaqueObject.queue_push.default, arg1_1, sin);  getitem = sin = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    with_effects_2 = torch.ops.higher_order.with_effects(getitem_2, torch.ops._TestOpaqueObject.queue_pop.default, arg1_1);  getitem_2 = None
    getitem_4 = with_effects_2[0]
    getitem_5 = with_effects_2[1];  with_effects_2 = None
    with_effects_3 = torch.ops.higher_order.with_effects(getitem_4, torch.ops._TestOpaqueObject.queue_size.default, arg1_1);  getitem_4 = None
    getitem_6 = with_effects_3[0];  with_effects_3 = None
    sub = torch.ops.aten.sub.Tensor(getitem_5, 1);  getitem_5 = None
    with_effects_4 = torch.ops.higher_order.with_effects(getitem_6, torch.ops._TestOpaqueObject.queue_pop.default, arg1_1);  getitem_6 = None
    getitem_8 = with_effects_4[0]
    getitem_9 = with_effects_4[1];  with_effects_4 = None
    with_effects_5 = torch.ops.higher_order.with_effects(getitem_8, torch.ops._TestOpaqueObject.queue_size.default, arg1_1);  getitem_8 = None
    getitem_10 = with_effects_5[0];  with_effects_5 = None
    add = torch.ops.aten.add.Tensor(getitem_9, 0);  getitem_9 = None
    return (getitem_10, sub, add, arg1_1)""",  # noqa: B950
        )
        _deregister_effectful_op("_TestOpaqueObject::queue_push.default")
        _deregister_effectful_op("_TestOpaqueObject::queue_pop.default")
        _deregister_effectful_op("_TestOpaqueObject::queue_size.default")


instantiate_parametrized_tests(TestOpaqueObject)

if __name__ == "__main__":
    run_tests()

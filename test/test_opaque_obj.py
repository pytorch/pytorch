# Owner(s): ["module: custom-operators"]
import copy

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._library.fake_class_registry import maybe_to_fake_obj
from torch._library.opaque_object import (
    get_payload,
    make_opaque,
    OpaqueType,
    set_payload,
)
from torch._subclasses.fake_tensor import FakeTensorMode
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
            # This is not accurate since the queue could have tensors that are
            # not rank 1
            ctx = torch._custom_op.impl.get_ctx()
            u0 = ctx.new_dynamic_size()
            return torch.empty(u0)

        self.lib._register_fake("queue_pop", pop_impl_fake)

        @torch.library.custom_op(
            "_TestOpaqueObject::queue_size",
            mutates_args=[],
        )
        def size_impl(q: OpaqueType) -> int:
            queue = get_payload(q)
            assert isinstance(queue, OpaqueQueue)
            return queue.size()

        @size_impl.register_fake
        def size_impl_fake(q: torch._C.ScriptObject) -> int:
            ctx = torch._custom_op.impl.get_ctx()
            u0 = ctx.new_dynamic_size()
            return u0

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

    def test_bad_fake(self):
        torch.library.define(
            "_TestOpaqueObject::bad_fake",
            "(__torch__.torch.classes.aten.OpaqueObject q, Tensor x) -> Tensor",
            lib=self.lib,
        )

        def f(q, x):
            torch.ops._TestOpaqueObject.bad_fake(q, x)
            return x.cos()

        def bad_fake1(q: torch._C.ScriptObject, b: torch.Tensor) -> torch.Tensor:
            payload = get_payload(q)
            return b * payload

        torch.library.register_fake(
            "_TestOpaqueObject::bad_fake", bad_fake1, lib=self.lib
        )

        with FakeTensorMode() as fake_mode:
            obj = make_opaque(1)
            fake_obj = maybe_to_fake_obj(fake_mode, obj)
            x = torch.ones(3)

            with self.assertRaisesRegex(
                ValueError,
                "get_payload: this function was called with a FakeScriptObject",
            ):
                torch.ops._TestOpaqueObject.bad_fake(fake_obj, x)

        def bad_fake2(q: torch._C.ScriptObject, b: torch.Tensor) -> torch.Tensor:
            set_payload(q, 2)
            return torch.empty_like(b)

        torch.library.register_fake(
            "_TestOpaqueObject::bad_fake", bad_fake2, lib=self.lib, allow_override=True
        )

        with FakeTensorMode() as fake_mode:
            obj = make_opaque(1)
            fake_obj = maybe_to_fake_obj(fake_mode, obj)
            x = torch.ones(3)

            with self.assertRaisesRegex(
                ValueError,
                "set_payload: this function was called with a FakeScriptObject",
            ):
                torch.ops._TestOpaqueObject.bad_fake(fake_obj, x)

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


instantiate_parametrized_tests(TestOpaqueObject)

if __name__ == "__main__":
    run_tests()

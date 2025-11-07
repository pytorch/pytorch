# Owner(s): ["module: custom-operators"]

import random

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import register_opaque_type
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


class RNGState:
    def __init__(self, seed):
        self.rng = random.Random(seed)


register_opaque_type(OpaqueQueue, "_TestOpaqueObject_OpaqueQueue")
register_opaque_type(RNGState, "_TestOpaqueObject_RNGState")


class TestOpaqueObject(TestCase):
    def setUp(self):
        self.lib = torch.library.Library("_TestOpaqueObject", "FRAGMENT")  # noqa: TOR901

        torch.library.define(
            "_TestOpaqueObject::queue_push",
            "(_TestOpaqueObject_OpaqueQueue a, Tensor b) -> ()",
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
            "queue_pop(_TestOpaqueObject_OpaqueQueue a) -> Tensor",
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
            "(Tensor x, _TestOpaqueObject_RNGState obj) -> Tensor",
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
            "(Tensor x, _TestOpaqueObject_RNGState obj) -> Tensor",
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


instantiate_parametrized_tests(TestOpaqueObject)


if __name__ == "__main__":
    run_tests()

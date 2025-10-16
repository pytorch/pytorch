# Owner(s): ["module: custom-operators"]

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._library.opaque_object import register_opaque_type


class OpaqueQueue:
    def __init__(self, queue: list[torch.Tensor], init_tensor_: torch.Tensor) -> None:
        super().__init__()
        self.queue = queue
        self.init_tensor_ = init_tensor_

    def push(self, tensor: torch.Tensor) -> None:
        self.queue.append(tensor)

    def pop(self) -> torch.Tensor:
        if len(self.queue) > 0:
            return self.queue.pop(0)
        return self.init_tensor_

    def size(self) -> int:
        return len(self.queue)


class TestOpaqueObject(TestCase):
    def setUp(self):
        self.lib = torch.library.Library("_TestOpaqueObject", "FRAGMENT")  # noqa: TOR901

        register_opaque_type(OpaqueQueue, "_TestOpaqueObject_OpaqueQueue")

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

        self.lib.define(
            "queue_pop(_TestOpaqueObject_OpaqueQueue a) -> Tensor",
        )

        def pop_impl(queue: OpaqueQueue) -> torch.Tensor:
            assert isinstance(queue, OpaqueQueue)
            return queue.pop()

        self.lib.impl("queue_pop", pop_impl, "CompositeExplicitAutograd")

        @torch.library.custom_op(
            "_TestOpaqueObject::queue_size",
            mutates_args=[],
        )
        def size_impl(queue: OpaqueQueue) -> int:
            assert isinstance(queue, OpaqueQueue)
            return queue.size()

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


if __name__ == "__main__":
    run_tests()

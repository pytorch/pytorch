# Owner(s): ["module: custom-operators"]

import torch
from torch._dynamo.test_case import run_tests, TestCase


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
        self.lib = torch.library.Library("_TestOpaqueObject", "FRAGMENT")

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
            queue = torch._C.OpaqueObject.get_payload(q)
            assert isinstance(queue, OpaqueQueue)
            queue.push(b)

        self.lib.define(
            "queue_pop(__torch__.torch.classes.aten.OpaqueObject a) -> Tensor",
        )

        def pop_impl(q: torch._C.ScriptObject) -> torch.Tensor:
            queue = torch._C.OpaqueObject.get_payload(q)
            assert isinstance(queue, OpaqueQueue)
            return queue.pop()

        self.lib.impl("queue_pop", pop_impl, "CompositeExplicitAutograd")

        super().setUp()

    def tearDown(self):
        self.lib._destroy()

        super().tearDown()

    def test_creation(self):
        queue = OpaqueQueue([], torch.zeros(3))
        obj = torch._C.OpaqueObject(queue)

        # obj.payload stores a direct reference to this python queue object
        self.assertEqual(obj.payload, queue)
        queue.push(torch.ones(3))
        self.assertEqual(obj.payload.size(), 1)

        boxed = obj.boxed()
        self.assertTrue(isinstance(boxed, torch._C.ScriptObject))

        unboxed = torch._C.OpaqueObject.unbox(boxed)
        self.assertTrue(isinstance(unboxed, torch._C.OpaqueObject))
        self.assertEqual(unboxed.payload, queue)

        payload = torch._C.OpaqueObject.get_payload(boxed)
        self.assertEqual(payload, queue)

    def test_ops(self):
        queue = OpaqueQueue([], torch.zeros(3))
        cpp_obj = torch._C.OpaqueObject(queue).boxed()

        torch.ops._TestOpaqueObject.queue_push(cpp_obj, torch.ones(3) + 1)
        self.assertEqual(queue.size(), 1)
        popped = torch.ops._TestOpaqueObject.queue_pop(cpp_obj)
        self.assertEqual(popped, torch.ones(3) + 1)
        self.assertEqual(queue.size(), 0)


if __name__ == "__main__":
    run_tests()

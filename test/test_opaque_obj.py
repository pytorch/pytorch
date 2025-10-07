# Owner(s): ["module: custom-operators"]
import copy

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._library.opaque_object import (
    get_payload,
    make_opaque,
    OpaqueType,
    set_payload,
)


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

        self.lib.define(
            "queue_pop(__torch__.torch.classes.aten.OpaqueObject a) -> Tensor",
        )

        def pop_impl(q: torch._C.ScriptObject) -> torch.Tensor:
            queue = get_payload(q)
            assert isinstance(queue, OpaqueQueue)
            return queue.pop()

        self.lib.impl("queue_pop", pop_impl, "CompositeExplicitAutograd")

        @torch.library.custom_op(
            "_TestOpaqueObject::queue_size",
            mutates_args=[],
        )
        def size_impl(q: OpaqueType) -> int:
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


if __name__ == "__main__":
    run_tests()

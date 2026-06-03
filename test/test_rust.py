# Owner(s): ["module: devx"]

import torch
import torch._rust
from torch.testing._internal.common_utils import run_tests, TestCase


class TestRustBindings(TestCase):
    def test_function(self):
        self.assertEqual(torch._rust.test_function(), "hello from rust")

    def test_tensor_size(self):
        t = torch.zeros(2, 3, 4)
        self.assertEqual(torch._rust.tensor_size(t), [2, 3, 4])

    def test_tensor_size_scalar(self):
        t = torch.tensor(0.0)
        self.assertEqual(torch._rust.tensor_size(t), [])

    def test_tensor_size_1d(self):
        t = torch.zeros(5)
        self.assertEqual(torch._rust.tensor_size(t), [5])

    def test_collect_tensors_single(self):
        t = torch.zeros(3)
        result = torch._rust.collect_tensors(t)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], t)
        self.assertIsInstance(result[0], torch.Tensor)

    def test_collect_tensors_non_tensor(self):
        self.assertEqual(torch._rust.collect_tensors(42), [])
        self.assertEqual(torch._rust.collect_tensors("hello"), [])
        self.assertEqual(torch._rust.collect_tensors(None), [])
        self.assertEqual(torch._rust.collect_tensors(b"bytes"), [])

    def test_collect_tensors_list(self):
        a, b = torch.zeros(2), torch.ones(3)
        result = torch._rust.collect_tensors([a, 1, b, "x"])
        self.assertEqual([id(t) for t in result], [id(a), id(b)])

    def test_collect_tensors_tuple(self):
        a, b = torch.zeros(2), torch.ones(3)
        result = torch._rust.collect_tensors((a, b))
        self.assertEqual([id(t) for t in result], [id(a), id(b)])

    def test_collect_tensors_dict_collects_values(self):
        a, b = torch.zeros(2), torch.ones(3)
        result = torch._rust.collect_tensors({"a": a, "b": b})
        self.assertEqual([id(t) for t in result], [id(a), id(b)])

    def test_collect_tensors_dict_keys_ignored(self):
        a, b = torch.zeros(2), torch.ones(3)
        result = torch._rust.collect_tensors({a: b})
        self.assertEqual([id(t) for t in result], [id(b)])

    def test_collect_tensors_set(self):
        a, b = torch.zeros(2), torch.ones(3)
        result = torch._rust.collect_tensors({a, b})
        self.assertEqual({id(t) for t in result}, {id(a), id(b)})

    def test_collect_tensors_frozenset(self):
        a, b = torch.zeros(2), torch.ones(3)
        result = torch._rust.collect_tensors(frozenset([a, b]))
        self.assertEqual({id(t) for t in result}, {id(a), id(b)})

    def test_collect_tensors_nested(self):
        a, b, c = torch.zeros(1), torch.ones(2), torch.zeros(3)
        result = torch._rust.collect_tensors([a, (b,), {"k": [c]}])
        self.assertEqual([id(t) for t in result], [id(a), id(b), id(c)])

    def test_collect_tensors_empty_containers(self):
        self.assertEqual(torch._rust.collect_tensors([]), [])
        self.assertEqual(torch._rust.collect_tensors(()), [])
        self.assertEqual(torch._rust.collect_tensors({}), [])
        self.assertEqual(torch._rust.collect_tensors(set()), [])


if __name__ == "__main__":
    run_tests()

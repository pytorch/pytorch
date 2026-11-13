# mypy: allow-untyped-defs
# Owner(s): ["module: unknown"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.data import SharedDict, SharedList
from torch.utils.data._shared_container import SharedTensor, to_shared_dataset


class TestSharedList(TestCase):
    def test_basic_types(self) -> None:
        sl = SharedList([1, 2, 3, 4, 5])
        self.assertEqual(len(sl), 5)
        self.assertEqual(sl[0], 1)
        self.assertEqual(sl[-1], 5)
        self.assertEqual(list(sl), [1, 2, 3, 4, 5])

    def test_slicing(self) -> None:
        sl = SharedList([1, 2, 3, 4, 5])
        self.assertEqual(sl[1:4], [2, 3, 4])
        self.assertEqual(sl[:3], [1, 2, 3])
        self.assertEqual(sl[2:], [3, 4, 5])

    def test_strings(self) -> None:
        sl = SharedList(["hello", "world", "foo", "bar"])
        self.assertEqual(sl[0], "hello")
        self.assertEqual(sl[1], "world")
        self.assertEqual(len(sl), 4)

    def test_nested_dicts(self) -> None:
        data = [{"a": 1, "b": [2, 3]}, {"c": 4}]
        sl = SharedList(data)
        self.assertEqual(sl[0], {"a": 1, "b": [2, 3]})
        self.assertEqual(sl[1], {"c": 4})

    def test_none_values(self) -> None:
        sl = SharedList([None, 1, None, "a"])
        self.assertIsNone(sl[0])
        self.assertEqual(sl[1], 1)
        self.assertIsNone(sl[2])
        self.assertEqual(sl[3], "a")

    def test_empty_list(self) -> None:
        sl = SharedList([])
        self.assertEqual(len(sl), 0)
        self.assertEqual(list(sl), [])

    def test_contains(self) -> None:
        sl = SharedList([1, "hello", None])
        self.assertIn(1, sl)
        self.assertIn("hello", sl)
        self.assertIn(None, sl)
        self.assertNotIn(999, sl)

    def test_index(self) -> None:
        sl = SharedList(["a", "b", "c", "b"])
        self.assertEqual(sl.index("a"), 0)
        self.assertEqual(sl.index("b"), 1)
        self.assertEqual(sl.index("b", 2), 3)

    def test_index_not_found(self) -> None:
        sl = SharedList([1, 2, 3])
        with self.assertRaises(ValueError):
            sl.index(999)

    def test_count(self) -> None:
        sl = SharedList([1, 2, 1, 1, 3])
        self.assertEqual(sl.count(1), 3)
        self.assertEqual(sl.count(2), 1)
        self.assertEqual(sl.count(999), 0)

    def test_copy(self) -> None:
        sl = SharedList([1, 2, 3])
        sl_copy = sl.copy()
        self.assertEqual(len(sl_copy), 3)
        self.assertEqual(list(sl_copy), [1, 2, 3])
        self.assertIsNot(sl, sl_copy)
        self.assertIsNot(sl._storage, sl_copy._storage)

    def test_from_sharedlist(self) -> None:
        sl1 = SharedList([1, 2, 3])
        sl2 = SharedList(sl1)
        self.assertEqual(list(sl2), [1, 2, 3])

    def test_iteration(self) -> None:
        sl = SharedList([1, 2, 3])
        self.assertEqual(list(sl), [1, 2, 3])

    def test_negative_index(self) -> None:
        sl = SharedList([1, 2, 3])
        self.assertEqual(sl[-1], 3)
        self.assertEqual(sl[-2], 2)

    def test_out_of_bounds(self) -> None:
        sl = SharedList([1, 2])
        with self.assertRaises(IndexError):
            _ = sl[5]
        with self.assertRaises(IndexError):
            _ = sl[-3]

    def test_large_dataset(self) -> None:
        N = 5000
        data = [{"id": i, "value": f"item_{i}_" * 20} for i in range(N)]
        sl = SharedList(data)
        self.assertEqual(len(sl), N)
        self.assertEqual(sl[0]["id"], 0)
        self.assertEqual(sl[N - 1]["id"], N - 1)
        self.assertEqual(sl[42]["value"], "item_42_" * 20)

    def test_tuple_keys_values(self) -> None:
        data = [{(1, 2): "a"}, {("x", "y"): (3, 4)}]
        sl = SharedList(data)
        self.assertEqual(sl[0], {(1, 2): "a"})
        self.assertEqual(sl[1], {("x", "y"): (3, 4)})

    def test_bytes_data(self) -> None:
        data = [b"hello", b"world", b"\x00\xff\x42"]
        sl = SharedList(data)
        self.assertEqual(sl[0], b"hello")
        self.assertEqual(sl[2], b"\x00\xff\x42")

    def test_slice_with_step(self) -> None:
        sl = SharedList([1, 2, 3, 4, 5, 6])
        self.assertEqual(sl[::2], [1, 3, 5])
        self.assertEqual(sl[1::2], [2, 4, 6])
        self.assertEqual(sl[::-1], [6, 5, 4, 3, 2, 1])

    def test_with_dataloader(self) -> None:
        from torch.utils.data import DataLoader, Dataset

        class MyDataset(Dataset):
            def __init__(self, data):
                self.data = SharedList(data)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        ds = MyDataset([1, 2, 3, 4])
        loader = DataLoader(ds, batch_size=2, num_workers=0)
        batches = list(loader)
        self.assertEqual(batches[0].tolist(), [1, 2])
        self.assertEqual(batches[1].tolist(), [3, 4])


class TestSharedDict(TestCase):
    def test_basic(self) -> None:
        sd = SharedDict({"a": 1, "b": "two", "c": 999})
        self.assertEqual(len(sd), 3)
        self.assertEqual(sd["a"], 1)
        self.assertEqual(sd["b"], "two")
        self.assertEqual(sd["c"], 999)

    def test_get_default(self) -> None:
        sd = SharedDict({"a": 1})
        self.assertEqual(sd.get("a"), 1)
        self.assertEqual(sd.get("x"), None)
        self.assertEqual(sd.get("x", "default"), "default")

    def test_contains(self) -> None:
        sd = SharedDict({"a": 1})
        self.assertIn("a", sd)
        self.assertNotIn("b", sd)

    def test_keys_values_items(self) -> None:
        sd = SharedDict({"a": 1, "b": 2, "c": 3})
        self.assertEqual(set(sd.keys()), {"a", "b", "c"})
        self.assertEqual(set(sd.values()), {1, 2, 3})
        items = list(sd.items())
        self.assertEqual(len(items), 3)

    def test_iter(self) -> None:
        sd = SharedDict({"a": 1, "b": 2})
        self.assertEqual(set(sd), {"a", "b"})

    def test_kwargs_init(self) -> None:
        sd = SharedDict(x=1, y=2)
        self.assertEqual(sd["x"], 1)
        self.assertEqual(sd["y"], 2)

    def test_combined_init(self) -> None:
        sd = SharedDict({"a": 1}, b=2)
        self.assertEqual(sd["a"], 1)
        self.assertEqual(sd["b"], 2)

    def test_empty(self) -> None:
        sd = SharedDict({})
        self.assertEqual(len(sd), 0)
        self.assertEqual(sd.keys(), [])
        self.assertEqual(sd.values(), [])
        self.assertEqual(sd.items(), [])

    def test_tuple_keys(self) -> None:
        sd = SharedDict({(1, 2): "tuple_key"})
        self.assertEqual(sd[(1, 2)], "tuple_key")

    def test_none_keys(self) -> None:
        sd = SharedDict({None: "none_value"})
        self.assertEqual(sd[None], "none_value")

    def test_key_not_found(self) -> None:
        sd = SharedDict({"a": 1})
        with self.assertRaises(KeyError):
            _ = sd["b"]

    def test_large_dict(self) -> None:
        N = 1000
        d = {f"key_{i}": f"value_{i}" for i in range(N)}
        sd = SharedDict(d)
        self.assertEqual(len(sd), N)
        self.assertEqual(sd["key_0"], "value_0")
        self.assertEqual(sd[f"key_{N - 1}"], f"value_{N - 1}")


class TestSharedTensor(TestCase):
    def test_basic(self) -> None:
        tensors = [torch.ones(5), torch.zeros(3), torch.randn(4)]
        st = SharedTensor(tensors)
        self.assertEqual(len(st), 3)
        self.assertTrue(torch.equal(st[0], torch.ones(5)))
        self.assertTrue(torch.equal(st[1], torch.zeros(3)))
        self.assertEqual(st[2].numel(), 4)

    def test_single_element(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0])
        st = SharedTensor([t])
        self.assertEqual(len(st), 1)
        self.assertTrue(torch.equal(st[0], t))

    def test_empty(self) -> None:
        st = SharedTensor([])
        self.assertEqual(len(st), 0)

    def test_different_dtypes(self) -> None:
        st = SharedTensor(
            [torch.ones(2, dtype=torch.float32), torch.ones(3, dtype=torch.float32)]
        )
        self.assertEqual(len(st), 2)

    def test_negative_index(self) -> None:
        tensors = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]
        st = SharedTensor(tensors)
        self.assertTrue(torch.equal(st[-1], torch.tensor([3.0])))

    def test_out_of_bounds(self) -> None:
        st = SharedTensor([torch.ones(2)])
        with self.assertRaises(IndexError):
            st[10]

    def test_iteration(self) -> None:
        tensors = [torch.ones(2), torch.zeros(2)]
        st = SharedTensor(tensors)
        result = list(st)
        self.assertEqual(len(result), 2)
        self.assertTrue(torch.equal(result[0], torch.ones(2)))


class TestToSharedDataset(TestCase):
    def test_converts_large_list(self) -> None:
        class DS:
            def __init__(self):
                self.data = list(range(10000))
                self.small = [1, 2, 3]

        ds = DS()
        to_shared_dataset(ds, threshold_bytes=100)
        self.assertIsInstance(ds.data, SharedList)
        self.assertIsInstance(ds.small, list)

    def test_converts_large_dict(self) -> None:
        class DS:
            def __init__(self):
                self.labels = {i: f"v{i}" for i in range(5000)}

        ds = DS()
        to_shared_dataset(ds, threshold_bytes=100)
        self.assertIsInstance(ds.labels, SharedDict)

    def test_preserves_non_data_attrs(self) -> None:
        class DS:
            def __init__(self):
                self.transform = lambda x: x
                self.data = list(range(10000))

        ds = DS()
        to_shared_dataset(ds, threshold_bytes=100)
        self.assertIsInstance(ds.data, SharedList)
        self.assertIsInstance(ds.transform, type(lambda: None))

    def test_skips_already_shared(self) -> None:
        orig = SharedList([1, 2, 3])

        class DS:
            def __init__(self):
                self.data = orig

        ds = DS()
        to_shared_dataset(ds, threshold_bytes=1)
        self.assertIs(ds.data, orig)

    def test_subclass_dataset(self) -> None:
        from torch.utils.data import Dataset

        class MyDS(Dataset):
            def __init__(self):
                self.paths = list(range(5000))

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                return self.paths[idx]

        ds = MyDS()
        to_shared_dataset(ds, threshold_bytes=100)
        self.assertIsInstance(ds.paths, SharedList)
        self.assertEqual(len(ds), 5000)


class TestDataLoaderSharedMemory(TestCase):
    def test_use_shared_memory_flag(self) -> None:
        from torch.utils.data import DataLoader, Dataset

        class DS(Dataset):
            def __init__(self):
                self.data = [f"item_{i}_" * 100 for i in range(5000)]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        dl = DataLoader(DS(), batch_size=4, use_shared_memory=True)
        self.assertIsInstance(dl.dataset.data, SharedList)

    def test_use_shared_memory_small(self) -> None:
        from torch.utils.data import DataLoader, Dataset

        class DS(Dataset):
            def __init__(self):
                self.data = [1]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        dl = DataLoader(DS(), batch_size=1, use_shared_memory=True)
        self.assertIsInstance(dl.dataset.data, list)
        items = [batch.item() for batch in dl]
        self.assertEqual(items, [1])


if __name__ == "__main__":
    run_tests()

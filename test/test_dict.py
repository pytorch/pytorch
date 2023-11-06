from __future__ import annotations

import json

import math

import os
import re

import numpy as np

import torch.cuda
from torch import multiprocessing as mp, nn
from torch.dict import pad, pad_sequence, TensorDict, TensorDictBase, TensorDictParams
from torch.dict.base import is_tensor_collection
from torch.dict.tensordict import _getitem_batch_size
from torch.dict.utils import assert_allclose_td, convert_ellipsis_to_idx
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TemporaryDirectoryName,
    TestCase,
)
from torch.utils._pytree import tree_map

TD_BATCH_SIZE = 4


def decompose(td):
    for v in td.values():
        if is_tensor_collection(v):
            yield from decompose(v)
        else:
            yield v


def get_available_devices():
    devices = [torch.device("cpu")]
    n_cuda = torch.cuda.device_count()
    if n_cuda > 0:
        for i in range(n_cuda):
            devices += [torch.device(f"cuda:{i}")]
    return devices


class TestTensorDicts(TestCase):
    @property
    def td_device(self):
        # A typical tensordict, on device
        if torch.cuda.device_count():
            device = "cuda:0"
        else:
            device = "cpu"
        return TensorDict(
            source={
                "a": torch.randn(4, 3, 2, 1, 5),
                "b": torch.randn(4, 3, 2, 1, 10),
                "c": torch.randint(10, (4, 3, 2, 1, 3)),
            },
            batch_size=[4, 3, 2, 1],
            device=device,
        )

    @property
    def td_memmap(self):
        return self.td_nested.memmap_()

    @property
    def td_no_device(self):
        # A typical tensordict, on device
        if torch.cuda.device_count():
            device = "cuda:0"
        else:
            device = "cpu"
        return TensorDict(
            source={
                "a": torch.randn(4, 3, 2, 1, 5, device=device),
                "b": torch.randn(4, 3, 2, 1, 10, device=device),
                "c": torch.randint(10, (4, 3, 2, 1, 3)),
            },
            batch_size=[4, 3, 2, 1],
        )

    @property
    def td_nested(self):
        # A typical tensordict, on device
        if torch.cuda.device_count():
            device = "cuda:0"
        else:
            device = "cpu"
        return TensorDict(
            source={
                "a": torch.randn(4, 3, 2, 1, 5, device=device),
                "b": torch.randn(4, 3, 2, 1, 10, device=device),
                "c": torch.randint(10, (4, 3, 2, 1, 3)),
                "d": TensorDict(
                    {"e": torch.randn(4, 3, 2, 1, 2)}, batch_size=[4, 3, 2, 1, 2]
                ),
            },
            batch_size=[4, 3, 2, 1],
        )

    @property
    def td_params(self):
        return TensorDictParams(self.td_nested)

    TD_TYPES = ["td_device", "td_no_device", "td_nested", "td_params", "td_memmap"]

    @parametrize("td_type", TD_TYPES)
    def test_creation(self, td_type):
        self.assertIsNot(getattr(self, td_type), None)

    @parametrize("td_type", TD_TYPES)
    def test_squeeze_unsqueeze(self, td_type):
        data = getattr(self, td_type)
        data_u = data.unsqueeze(-1)
        self.assertEqual(data_u.shape, torch.Size([4, 3, 2, 1, 1]))
        self.assertEqual(data_u.squeeze().shape, torch.Size([4, 3, 2]))
        self.assertEqual(data_u.squeeze(0).shape, torch.Size([4, 3, 2, 1, 1]))
        self.assertEqual(data_u.squeeze(-1).shape, torch.Size([4, 3, 2, 1]))
        data_u = data.unsqueeze(-3)
        self.assertEqual(data_u.shape, torch.Size([4, 3, 1, 2, 1]))
        self.assertEqual(data_u.squeeze().shape, torch.Size([4, 3, 2]))
        self.assertEqual(data_u.squeeze(0).shape, torch.Size([4, 3, 1, 2, 1]))
        self.assertEqual(data_u.squeeze(-3).shape, torch.Size([4, 3, 2, 1]))
        data_u = data.unsqueeze(0)
        self.assertEqual(data_u.shape, torch.Size([1, 4, 3, 2, 1]))
        self.assertEqual(data_u.squeeze().shape, torch.Size([4, 3, 2]))
        self.assertEqual(data_u.squeeze(0).shape, torch.Size([4, 3, 2, 1]))
        self.assertEqual(data_u.squeeze(-3).shape, torch.Size([1, 4, 3, 2, 1]))
        data_u = data.unsqueeze(2)
        self.assertEqual(data_u.shape, torch.Size([4, 3, 1, 2, 1]))
        self.assertEqual(data_u.squeeze().shape, torch.Size([4, 3, 2]))
        self.assertEqual(data_u.squeeze(0).shape, torch.Size([4, 3, 1, 2, 1]))
        self.assertEqual(data_u.squeeze(2).shape, torch.Size([4, 3, 2, 1]))
        for item in data_u.values(include_nested=True):
            assert item.shape[:5] == torch.Size([4, 3, 1, 2, 1])

    def test_batchsize_reset(self):
        td = TensorDict(
            {"a": torch.randn(3, 4, 5, 6), "b": torch.randn(3, 4, 5)}, batch_size=[3, 4]
        )
        # smoke-test
        td.batch_size = torch.Size([3])

        # test with list
        td.batch_size = [3]

        # test with tuple
        td.batch_size = (3,)

        # incompatible size
        with self.assertRaisesRegex(
            RuntimeError,
            expected_regex=re.escape(
                "the tensor a has shape torch.Size([3, 4, 5, 6]) which is incompatible with the batch-size torch.Size([3, 5])"
            ),
        ):
            td.batch_size = [3, 5]

        # test set
        td.set("c", torch.randn(3))

        # test index
        td[torch.tensor([1, 2])]
        td[:]
        td[[1, 2]]
        with self.assertRaisesRegex(
            IndexError,
            expected_regex="too many indices for tensor of dimension 1",
        ):
            td[:, 0]

        # test a greater batch_size
        td = TensorDict(
            {"a": torch.randn(3, 4, 5, 6), "b": torch.randn(3, 4, 5)}, batch_size=[3, 4]
        )
        td.batch_size = torch.Size([3, 4, 5])

        td.set("c", torch.randn(3, 4, 5, 6))
        with self.assertRaisesRegex(
            RuntimeError,
            expected_regex=re.escape(
                "batch dimension mismatch, got self.batch_size=torch.Size([3, 4, 5]) and value.shape=torch.Size([3, 4, 2])"
            ),
        ):
            td.set("d", torch.randn(3, 4, 2))

    @parametrize("device", get_available_devices())
    def test_cat_td(self, device):
        torch.manual_seed(1)
        d = {
            "key1": torch.randn(4, 5, 6, device=device),
            "key2": torch.randn(4, 5, 10, device=device),
            "key3": {"key4": torch.randn(4, 5, 10, device=device)},
        }
        td1 = TensorDict(batch_size=(4, 5), source=d, device=device)
        d = {
            "key1": torch.randn(4, 10, 6, device=device),
            "key2": torch.randn(4, 10, 10, device=device),
            "key3": {"key4": torch.randn(4, 10, 10, device=device)},
        }
        td2 = TensorDict(batch_size=(4, 10), source=d, device=device)

        td_cat = torch.cat([td1, td2], 1)
        assert td_cat.batch_size == torch.Size([4, 15])
        d = {
            "key1": torch.zeros(4, 15, 6, device=device),
            "key2": torch.zeros(4, 15, 10, device=device),
            "key3": {"key4": torch.zeros(4, 15, 10, device=device)},
        }
        td_out = TensorDict(batch_size=(4, 15), source=d, device=device)
        data_ptr_set_before = {val.data_ptr() for val in decompose(td_out)}
        torch.cat([td1, td2], 1, out=td_out)
        data_ptr_set_after = {val.data_ptr() for val in decompose(td_out)}
        self.assertEqual(data_ptr_set_before, data_ptr_set_after)
        self.assertEqual(td_out.batch_size, torch.Size([4, 15]))
        assert (td_out["key1"] != 0).all()
        assert (td_out["key2"] != 0).all()
        assert (td_out["key3", "key4"] != 0).all()

    def test_create_on_device(self):
        if not torch.cuda.device_count():
            self.skipTest("no cuda")
        device = torch.device(0)

        # TensorDict
        td = TensorDict({}, [5])
        assert td.device is None

        td.set("a", torch.randn(5, device=device))
        assert td.device is None

        td = TensorDict({}, [5], device="cuda:0")
        td.set("a", torch.randn(5, 1))
        assert td.get("a").device == device

        # TensorDict, indexed
        td = TensorDict({}, [5])
        subtd = td[1]
        assert subtd.device is None

        subtd.set("a", torch.randn(1, device=device))
        # setting element of subtensordict doesn't set top-level device
        assert subtd.device is None

        subtd = subtd.to(device)
        assert subtd.device == device
        assert subtd["a"].device == device

        td = TensorDict({}, [5], device="cuda:0")
        subtd = td[1]
        subtd.set("a", torch.randn(1))
        assert subtd.get("a").device == device

        td = TensorDict({}, [5], device="cuda:0")
        subtd = td[1:3]
        subtd.set("a", torch.randn(2))
        assert subtd.get("a").device == device

    def test_empty(self):
        td = TensorDict(
            {
                "a": torch.zeros(()),
                ("b", "c"): torch.zeros(()),
                ("b", "d", "e"): torch.zeros(()),
            },
            [],
        )
        td_empty = td.empty(recurse=False)
        assert len(list(td_empty.keys())) == 0
        td_empty = td.empty(recurse=True)
        assert len(list(td_empty.keys())) == 1
        assert len(list(td_empty.get("b").keys())) == 1

    def test_error_on_contains(self):
        td = TensorDict(
            {"a": TensorDict({"b": torch.rand(1, 2)}, [1, 2]), "c": torch.rand(1)}, [1]
        )
        with self.assertRaisesRegex(
            NotImplementedError,
            expected_regex="TensorDict does not support membership checks with the `in` keyword",
        ):
            "random_string" in td  # noqa: B015

    @parametrize("method", ["share_memory", "memmap"])
    def test_memory_lock(self, method):
        torch.manual_seed(1)
        td = TensorDict({"a": torch.randn(4, 5)}, batch_size=(4, 5))

        # lock=True
        if method == "share_memory":
            td.share_memory_()
        elif method == "memmap":
            td.memmap_()
        else:
            raise NotImplementedError

        td.set("a", torch.randn(4, 5), inplace=True)
        td.set_("a", torch.randn(4, 5))  # No exception because set_ ignores the lock

        with self.assertRaisesRegex(
            RuntimeError, expected_regex="Cannot modify locked TensorDict"
        ):
            td.set("a", torch.randn(4, 5))

        with self.assertRaisesRegex(
            RuntimeError, expected_regex="Cannot modify locked TensorDict"
        ):
            td.set("b", torch.randn(4, 5))

        with self.assertRaisesRegex(
            RuntimeError, expected_regex="Cannot modify locked TensorDict"
        ):
            td.set("b", torch.randn(4, 5), inplace=True)

    @parametrize("inplace", [True, False])
    def test_exclude_nested(self, inplace):
        tensor_1 = torch.rand(4, 5, 6, 7)
        tensor_2 = torch.rand(4, 5, 6, 7)
        sub_sub_tensordict = TensorDict(
            {"t1": tensor_1, "t2": tensor_2}, batch_size=[4, 5, 6]
        )
        sub_tensordict = TensorDict(
            {"double_nested": sub_sub_tensordict}, batch_size=[4, 5]
        )
        tensordict = TensorDict(
            {
                "a": torch.rand(4, 3),
                "b": torch.rand(4, 2),
                "c": torch.rand(4, 1),
                "nested": sub_tensordict,
            },
            batch_size=[4],
        )
        # making a copy for inplace tests
        tensordict2 = tensordict.clone()

        if inplace and tensordict.is_locked:
            with self.assertRaisesRegex(
                RuntimeError,
                expected_regex=re.escape(
                    "Cannot modify locked TensorDict. For in-place modification, consider using the `set_()` method and make sure the key is present."
                ),
            ):
                excluded = tensordict.exclude(
                    "b", ("nested", "double_nested", "t2"), inplace=inplace
                )
            return

        excluded = tensordict.exclude(
            "b", ("nested", "double_nested", "t2"), inplace=inplace
        )

        assert set(excluded.keys(include_nested=True)) == {
            "a",
            "c",
            "nested",
            ("nested", "double_nested"),
            ("nested", "double_nested", "t1"),
        }

        if inplace:
            assert excluded is tensordict
            assert set(tensordict.keys(include_nested=True)) == {
                "a",
                "c",
                "nested",
                ("nested", "double_nested"),
                ("nested", "double_nested", "t1"),
            }
        else:
            assert excluded is not tensordict
            assert set(tensordict.keys(include_nested=True)) == {
                "a",
                "b",
                "c",
                "nested",
                ("nested", "double_nested"),
                ("nested", "double_nested", "t1"),
                ("nested", "double_nested", "t2"),
            }

        # excluding "nested" should exclude all subkeys also
        excluded2 = tensordict2.exclude("nested", inplace=inplace)
        assert set(excluded2.keys(include_nested=True)) == {"a", "b", "c"}

    @parametrize("device", get_available_devices())
    def test_expand(self, device):
        torch.manual_seed(1)
        d = {
            "key1": torch.randn(4, 5, 6, device=device),
            "key2": torch.randn(4, 5, 10, device=device),
        }
        td1 = TensorDict(batch_size=(4, 5), source=d)
        td2 = td1.expand(3, 7, 4, 5)
        assert td2.batch_size == torch.Size([3, 7, 4, 5])
        assert td2.get("key1").shape == torch.Size([3, 7, 4, 5, 6])
        assert td2.get("key2").shape == torch.Size([3, 7, 4, 5, 10])

    @parametrize("device", get_available_devices())
    def test_expand_with_singleton(self, device):
        torch.manual_seed(1)
        d = {
            "key1": torch.randn(1, 5, 6, device=device),
            "key2": torch.randn(1, 5, 10, device=device),
        }
        td1 = TensorDict(batch_size=(1, 5), source=d)
        td2 = td1.expand(3, 7, 4, 5)
        assert td2.batch_size == torch.Size([3, 7, 4, 5])
        assert td2.get("key1").shape == torch.Size([3, 7, 4, 5, 6])
        assert td2.get("key2").shape == torch.Size([3, 7, 4, 5, 10])

    @parametrize("device", get_available_devices())
    def test_filling_empty_tensordict(self, device):
        td = TensorDict({}, batch_size=[16], device=device)

        for i in range(16):
            other_td = TensorDict({"a": torch.randn(10), "b": torch.ones(1)}, [])
            td[i] = other_td

        assert td.device == device
        assert td.get("a").device == device
        assert (td.get("b") == 1).all()

    @parametrize("inplace", [True, False])
    @parametrize("separator", [",", "-"])
    def test_flatten_unflatten_key_collision(self, inplace, separator):
        td1 = TensorDict(
            {
                f"a{separator}b{separator}c": torch.zeros(3),
                "a": {"b": {"c": torch.zeros(3)}},
            },
            [],
        )
        td2 = TensorDict(
            {
                f"a{separator}b": torch.zeros(3),
                "a": {"b": torch.zeros(3)},
                "g": {"d": torch.zeros(3)},
            },
            [],
        )
        td3 = TensorDict(
            {
                f"a{separator}b{separator}c": torch.zeros(3),
                "a": {"b": {"c": torch.zeros(3), "d": torch.zeros(3)}},
            },
            [],
        )

        td4 = TensorDict(
            {
                f"a{separator}b{separator}c{separator}d": torch.zeros(3),
                "a": {"b": {"c": torch.zeros(3)}},
            },
            [],
        )

        td5 = TensorDict(
            {f"a{separator}b": torch.zeros(3), "a": {"b": {"c": torch.zeros(3)}}}, []
        )

        with self.assertRaisesRegex(
            KeyError,
            expected_regex=re.escape(
                "Flattening keys in tensordict causes keys [('a', 'b', 'c')] to collide."
            ),
        ):
            td1.flatten_keys(separator)

        with self.assertRaisesRegex(
            KeyError,
            expected_regex=re.escape(
                "Flattening keys in tensordict causes keys [('a', 'b')] to collide."
            ),
        ):
            td2.flatten_keys(separator)

        with self.assertRaisesRegex(
            KeyError,
            expected_regex=re.escape(
                "Flattening keys in tensordict causes keys [('a', 'b', 'c')] to collide."
            ),
        ):
            td3.flatten_keys(separator)

        with self.assertRaisesRegex(
            KeyError,
            expected_regex=re.escape(
                "Unflattening key(s) in tensordict will override an existing for unflattened key"
            ),
        ):
            td1.unflatten_keys(separator)

        with self.assertRaisesRegex(
            KeyError,
            expected_regex=re.escape(
                "Unflattening key(s) in tensordict will override an existing for unflattened key"
            ),
        ):
            td2.unflatten_keys(separator)

        with self.assertRaisesRegex(
            KeyError,
            expected_regex=re.escape(
                "Unflattening key(s) in tensordict will override an existing for unflattened key"
            ),
        ):
            td3.unflatten_keys(separator)

        with self.assertRaisesRegex(
            KeyError,
            expected_regex=re.escape(
                "Unflattening key(s) in tensordict will override an existing for unflattened key"
            ),
        ):
            td4.unflatten_keys(separator)

        with self.assertRaisesRegex(
            KeyError,
            expected_regex=re.escape(
                "Unflattening key(s) in tensordict will override an existing for unflattened key"
            ),
        ):
            td5.unflatten_keys(separator)

        td4_flat = td4.flatten_keys(separator)
        assert (f"a{separator}b{separator}c{separator}d") in td4_flat.keys()
        assert (f"a{separator}b{separator}c") in td4_flat.keys()

        td5_flat = td5.flatten_keys(separator)
        assert (f"a{separator}b") in td5_flat.keys()
        assert (f"a{separator}b{separator}c") in td5_flat.keys()

    @parametrize("batch_size", [None, [3, 4]])
    @parametrize("batch_dims", [None, 1, 2])
    @parametrize("device", get_available_devices())
    def test_from_dict(self, batch_size, batch_dims, device):
        data = {
            "a": torch.zeros(3, 4, 5),
            "b": {"c": torch.zeros(3, 4, 5, 6)},
            ("d", "e"): torch.ones(3, 4, 5),
            ("b", "f"): torch.zeros(3, 4, 5, 5),
            ("d", "g", "h"): torch.ones(3, 4, 5),
        }
        if batch_dims and batch_size:
            with self.assertRaisesRegex(
                ValueError, expected_regex="Cannot pass both batch_size and batch_dims"
            ):
                TensorDict.from_dict(
                    data, batch_size=batch_size, batch_dims=batch_dims, device=device
                )
            return
        data = TensorDict.from_dict(
            data, batch_size=batch_size, batch_dims=batch_dims, device=device
        )
        assert data.device == device
        assert "a" in data.keys()
        assert ("b", "c") in data.keys(True)
        assert ("b", "f") in data.keys(True)
        assert ("d", "e") in data.keys(True)
        assert data.device == device
        if batch_dims:
            assert data.ndim == batch_dims
            assert data["b"].ndim == batch_dims
            assert data["d"].ndim == batch_dims
            assert data["d", "g"].ndim == batch_dims
        elif batch_size:
            assert data.batch_size == torch.Size(batch_size)
            assert data["b"].batch_size == torch.Size(batch_size)
            assert data["d"].batch_size == torch.Size(batch_size)
            assert data["d", "g"].batch_size == torch.Size(batch_size)

    @parametrize("memmap", [True, False])
    @parametrize("params", [False, True])
    def test_from_module(self, memmap, params):
        net = nn.Transformer(
            d_model=16,
            nhead=2,
            num_encoder_layers=3,
            dim_feedforward=12,
        )
        td = TensorDict.from_module(net, as_module=params)
        # check that we have empty tensordicts, reflecting modules wihout params
        for subtd in td.values(True):
            if isinstance(subtd, TensorDictBase) and subtd.is_empty():
                break
        else:
            raise RuntimeError
        if memmap:
            td = td.detach().memmap_()
        net.load_state_dict(td.flatten_keys("."))

        if not memmap and params:
            assert set(td.parameters()) == set(net.parameters())

    def test_getitem_nested(self):
        tensor = torch.randn(4, 5, 6, 7)
        sub_sub_tensordict = TensorDict({"c": tensor}, [4, 5, 6])
        sub_tensordict = TensorDict({}, [4, 5])
        tensordict = TensorDict({}, [4])

        sub_tensordict["b"] = sub_sub_tensordict
        tensordict["a"] = sub_tensordict

        # check that content match
        assert (tensordict["a"] == sub_tensordict).all()
        assert (tensordict["a", "b"] == sub_sub_tensordict).all()
        assert (tensordict["a", "b", "c"] == tensor).all()

        # check that get method returns same contents
        assert (tensordict.get("a") == sub_tensordict).all()
        assert (tensordict.get(("a", "b")) == sub_sub_tensordict).all()
        assert (tensordict.get(("a", "b", "c")) == tensor).all()

        # check that shapes are kept
        assert tensordict.shape == torch.Size([4])
        assert sub_tensordict.shape == torch.Size([4, 5])
        assert sub_sub_tensordict.shape == torch.Size([4, 5, 6])

    @parametrize("td_type", ["td_device", "td_no_device", "td_nested", "td_params"])
    def test_inferred_view_size(self, td_type):
        td = getattr(self, td_type)
        self.assertIs(td.view(-1, 3, 2, 1), td)
        self.assertIs(td.view(4, -1, 2, 1), td)
        self.assertIs(td.view(4, 3, 2, 1), td)
        self.assertEqual(td.view(-1, 24).shape, torch.Size([1, 24]))

    def test_keys_view(self):
        tensor = torch.randn(4, 5, 6, 7)
        sub_sub_tensordict = TensorDict({"c": tensor}, [4, 5, 6])
        sub_tensordict = TensorDict({}, [4, 5])
        tensordict = TensorDict({}, [4])

        sub_tensordict["b"] = sub_sub_tensordict
        tensordict["a"] = sub_tensordict

        assert "a" in tensordict.keys()
        assert "random_string" not in tensordict.keys()

        assert ("a",) in tensordict.keys(include_nested=True)
        assert ("a", "b", "c") in tensordict.keys(include_nested=True)
        assert ("a", "c", "b") not in tensordict.keys(include_nested=True)

        with self.assertRaisesRegex(
            TypeError,
            expected_regex=re.escape(
                "Nested membership checks with tuples of strings is only supported when setting"
            ),
        ):
            ("a", "b", "c") in tensordict.keys()  # noqa: B015

        with self.assertRaisesRegex(
            TypeError, expected_regex="TensorDict keys are always strings."
        ):
            42 in tensordict.keys()  # noqa: B015

        with self.assertRaisesRegex(
            TypeError, expected_regex="TensorDict keys are always strings."
        ):
            ("a", 42) in tensordict.keys()  # noqa: B015

        keys = set(tensordict.keys())
        keys_nested = set(tensordict.keys(include_nested=True))

        assert keys == {"a"}
        assert keys_nested == {"a", ("a", "b"), ("a", "b", "c")}

        leaves = set(tensordict.keys(leaves_only=True))
        leaves_nested = set(tensordict.keys(include_nested=True, leaves_only=True))

        assert leaves == set()
        assert leaves_nested == {("a", "b", "c")}

    @parametrize("device", get_available_devices())
    def test_mask_td(self, device):
        torch.manual_seed(1)
        d = {
            "key1": torch.randn(4, 5, 6, device=device),
            "key2": torch.randn(4, 5, 10, device=device),
        }
        mask = torch.zeros(4, 5, dtype=torch.bool, device=device).bernoulli_()
        td = TensorDict(batch_size=(4, 5), source=d)

        td_masked = torch.masked_select(td, mask)
        assert len(td_masked.get("key1")) == td_masked.shape[0]

    @parametrize("device", get_available_devices())
    def test_memmap_as_tensor(self, device):
        td = TensorDict(
            {"a": torch.randn(3, 4), "b": {"c": torch.randn(3, 4)}},
            [3, 4],
            device="cpu",
        )
        td_memmap = td.clone().memmap_()
        assert (td == td_memmap).all()

        assert (td == td_memmap.apply(lambda x: x.clone())).all()
        if device.type == "cuda":
            td = td.pin_memory()
            td_memmap = td.clone().memmap_()
            td_memmap_pm = td_memmap.apply(lambda x: x.clone()).pin_memory()
            assert (td.pin_memory().to(device) == td_memmap_pm.to(device)).all()

    @parametrize("method", ["share_memory", "memmap"])
    def test_memory_lock(self, method):
        torch.manual_seed(1)
        td = TensorDict({"a": torch.randn(4, 5)}, batch_size=(4, 5))

        # lock=True
        if method == "share_memory":
            td.share_memory_()
        elif method == "memmap":
            td.memmap_()
        else:
            raise NotImplementedError

        td.set("a", torch.randn(4, 5), inplace=True)
        td.set_("a", torch.randn(4, 5))  # No exception because set_ ignores the lock

        with self.assertRaisesRegex(
            RuntimeError, expected_regex="Cannot modify locked TensorDict"
        ):
            td.set("a", torch.randn(4, 5))

        with self.assertRaisesRegex(
            RuntimeError, expected_regex="Cannot modify locked TensorDict"
        ):
            td.set("b", torch.randn(4, 5))

        with self.assertRaisesRegex(
            RuntimeError, expected_regex="Cannot modify locked TensorDict"
        ):
            td.set("b", torch.randn(4, 5), inplace=True)

    def test_pad(self):
        dim0_left, dim0_right, dim1_left, dim1_right = [0, 1, 0, 2]
        td = TensorDict(
            {
                "a": torch.ones(3, 4, 1),
                "b": torch.zeros(3, 4, 1, 1),
            },
            batch_size=[3, 4],
        )

        padded_td = pad(td, [dim0_left, dim0_right, dim1_left, dim1_right], value=0.0)

        expected_a = torch.cat([torch.ones(3, 4, 1), torch.zeros(1, 4, 1)], dim=0)
        expected_a = torch.cat([expected_a, torch.zeros(4, 2, 1)], dim=1)

        assert padded_td["a"].shape == (4, 6, 1)
        assert padded_td["b"].shape == (4, 6, 1, 1)
        assert torch.equal(padded_td["a"], expected_a)
        padded_td._check_batch_size()

    @parametrize("batch_first", [True, False])
    @parametrize("make_mask", [True, False])
    def test_pad_sequence(self, batch_first, make_mask):
        list_td = [
            TensorDict({"a": torch.ones((2,)), ("b", "c"): torch.ones((2, 3))}, [2]),
            TensorDict({"a": torch.ones((4,)), ("b", "c"): torch.ones((4, 3))}, [4]),
        ]
        padded_td = pad_sequence(
            list_td, batch_first=batch_first, return_mask=make_mask
        )
        if batch_first:
            assert padded_td.shape == torch.Size([2, 4])
            assert padded_td["a"].shape == torch.Size([2, 4])
            assert padded_td["a"][0, -1] == 0
            assert padded_td["b", "c"].shape == torch.Size([2, 4, 3])
            assert padded_td["b", "c"][0, -1, 0] == 0
        else:
            assert padded_td.shape == torch.Size([4, 2])
            assert padded_td["a"].shape == torch.Size([4, 2])
            assert padded_td["a"][-1, 0] == 0
            assert padded_td["b", "c"].shape == torch.Size([4, 2, 3])
            assert padded_td["b", "c"][-1, 0, 0] == 0
        if make_mask:
            assert "mask" in padded_td.keys()
            assert not padded_td["mask"].all()
        else:
            assert "mask" not in padded_td.keys()

    @parametrize("device", get_available_devices())
    def test_permute(self, device):
        torch.manual_seed(1)
        d = {
            "a": torch.randn(4, 5, 6, 9, device=device),
            "b": torch.randn(4, 5, 6, 7, device=device),
            "c": torch.randn(4, 5, 6, device=device),
        }
        td1 = TensorDict(batch_size=(4, 5, 6), source=d)
        td2 = torch.permute(td1, dims=(2, 1, 0))
        assert td2.shape == torch.Size((6, 5, 4))
        assert td2["a"].shape == torch.Size((6, 5, 4, 9))

        td2 = torch.permute(td1, dims=(-1, -3, -2))
        assert td2.shape == torch.Size((6, 4, 5))
        assert td2["c"].shape == torch.Size((6, 4, 5))

        td2 = torch.permute(td1, dims=(0, 1, 2))
        assert td2["a"].shape == torch.Size((4, 5, 6, 9))

    @parametrize("device", get_available_devices())
    def test_permute_exceptions(self, device):
        torch.manual_seed(1)
        d = {
            "a": torch.randn(4, 5, 6, 7, device=device),
            "b": torch.randn(4, 5, 6, 8, 9, device=device),
        }
        td1 = TensorDict(batch_size=(4, 5, 6), source=d)

        with self.assertRaises(ValueError):
            td2 = td1.permute(1, 1, 0)
            _ = td2.shape

        with self.assertRaises(ValueError):
            td2 = td1.permute(3, 2, 1, 0)
            _ = td2.shape

        with self.assertRaises(ValueError):
            td2 = td1.permute(2, -1, 0)
            _ = td2.shape

        with self.assertRaises(ValueError):
            td2 = td1.permute(2, 3, 0)
            _ = td2.shape

        with self.assertRaises(ValueError):
            td2 = td1.permute(2, -4, 0)
            _ = td2.shape

        with self.assertRaises(ValueError):
            td2 = td1.permute(2, 1)
            _ = td2.shape

    @parametrize("device", get_available_devices())
    def test_permute_with_tensordict_operations(self, device):
        torch.manual_seed(1)
        d = {
            "a": torch.randn(20, 6, 9, device=device),
            "b": torch.randn(20, 6, 7, device=device),
            "c": torch.randn(20, 6, device=device),
        }
        td1 = TensorDict(batch_size=(20, 6), source=d).view(4, 5, 6).permute(2, 1, 0)
        assert td1.shape == torch.Size((6, 5, 4))

        d = {
            "a": torch.randn(4, 5, 6, 7, 9, device=device),
            "b": torch.randn(4, 5, 6, 7, 7, device=device),
            "c": torch.randn(4, 5, 6, 7, device=device),
        }
        td1 = TensorDict(batch_size=(4, 5, 6, 7), source=d)[
            :, :, :, torch.tensor([1, 2])
        ].permute(3, 2, 1, 0)
        assert td1.shape == torch.Size((2, 6, 5, 4))

    @parametrize("device", get_available_devices())
    def test_requires_grad(self, device):
        torch.manual_seed(1)
        # Just one of the tensors have requires_grad
        tensordicts = [
            TensorDict(
                batch_size=[11, 12],
                source={
                    "key1": torch.randn(
                        11,
                        12,
                        5,
                        device=device,
                        requires_grad=True if i == 5 else False,
                    ),
                },
            )
            for i in range(10)
        ]
        stacked_td = torch.stack(tensordicts, 0)
        # First stacked tensor has requires_grad == True
        assert list(stacked_td.values())[0].requires_grad is True

    @parametrize("inplace", [True, False])
    def test_select_nested(self, inplace):
        tensor_1 = torch.rand(4, 5, 6, 7)
        tensor_2 = torch.rand(4, 5, 6, 7)
        sub_sub_tensordict = TensorDict(
            {"t1": tensor_1, "t2": tensor_2}, batch_size=[4, 5, 6]
        )
        sub_tensordict = TensorDict(
            {"double_nested": sub_sub_tensordict}, batch_size=[4, 5]
        )
        tensordict = TensorDict(
            {
                "a": torch.rand(4, 3),
                "b": torch.rand(4, 2),
                "c": torch.rand(4, 1),
                "nested": sub_tensordict,
            },
            batch_size=[4],
        )

        selected = tensordict.select(
            "b", ("nested", "double_nested", "t2"), inplace=inplace
        )

        assert set(selected.keys(include_nested=True)) == {
            "b",
            "nested",
            ("nested", "double_nested"),
            ("nested", "double_nested", "t2"),
        }

        if inplace:
            assert selected is tensordict
            assert set(tensordict.keys(include_nested=True)) == {
                "b",
                "nested",
                ("nested", "double_nested"),
                ("nested", "double_nested", "t2"),
            }
        else:
            assert selected is not tensordict
            assert set(tensordict.keys(include_nested=True)) == {
                "a",
                "b",
                "c",
                "nested",
                ("nested", "double_nested"),
                ("nested", "double_nested", "t1"),
                ("nested", "double_nested", "t2"),
            }

    def test_select_nested_missing(self):
        # checks that we keep a nested key even if missing nested keys are present
        td = TensorDict({"a": {"b": [1], "c": [2]}}, [])

        td_select = td.select(("a", "b"), "r", ("a", "z"), strict=False)
        assert ("a", "b") in list(td_select.keys(True, True))
        assert ("a", "b") in td_select.keys(True, True)

    def test_set_nested_keys(self):
        tensor = torch.randn(4, 5, 6, 7)
        tensor2 = torch.ones(4, 5, 6, 7)
        tensordict = TensorDict({}, [4])
        sub_tensordict = TensorDict({}, [4, 5])
        sub_sub_tensordict = TensorDict({"c": tensor}, [4, 5, 6])
        sub_sub_tensordict2 = TensorDict({"c": tensor2}, [4, 5, 6])
        sub_tensordict.set("b", sub_sub_tensordict)
        tensordict.set("a", sub_tensordict)
        assert tensordict.get(("a", "b")) is sub_sub_tensordict

        tensordict.set(("a", "b"), sub_sub_tensordict2)
        assert tensordict.get(("a", "b")) is sub_sub_tensordict2
        assert (tensordict.get(("a", "b", "c")) == 1).all()

    @parametrize("index0", [None, slice(None)])
    def test_set_sub_key(self, index0):
        # tests that parent tensordict is affected when subtensordict is set with a new key
        batch_size = [10, 10]
        source = {"a": torch.randn(10, 10, 10), "b": torch.ones(10, 10, 2)}
        td = TensorDict(source, batch_size=batch_size)
        idx0 = (index0, 0) if index0 is not None else 0
        td0 = td._get_sub_tensordict(idx0)
        idx = (index0, slice(2, 4)) if index0 is not None else slice(2, 4)
        sub_td = td._get_sub_tensordict(idx)
        if index0 is None:
            c = torch.randn(2, 10, 10)
        else:
            c = torch.randn(10, 2, 10)
        sub_td.set("c", c)
        assert (td.get("c")[idx] == sub_td.get("c")).all()
        assert (sub_td.get("c") == c).all()
        assert (td.get("c")[idx0] == 0).all()
        assert (td._get_sub_tensordict(idx0).get("c") == 0).all()
        assert (td0.get("c") == 0).all()

    def test_setdefault_nested(self):
        tensor = torch.randn(4, 5, 6, 7)
        tensor2 = torch.ones(4, 5, 6, 7)
        sub_sub_tensordict = TensorDict({"c": tensor}, [4, 5, 6])
        sub_tensordict = TensorDict({"b": sub_sub_tensordict}, [4, 5])
        tensordict = TensorDict({"a": sub_tensordict}, [4])

        # if key exists we return the existing value
        assert tensordict.setdefault(("a", "b", "c"), tensor2) is tensor

        assert tensordict.setdefault(("a", "b", "d"), tensor2) is tensor2
        assert (tensordict["a", "b", "d"] == 1).all()
        assert tensordict.get(("a", "b", "d")) is tensor2

    def test_setitem_nested(self):
        tensor = torch.randn(4, 5, 6, 7)
        tensor2 = torch.ones(4, 5, 6, 7)
        tensordict = TensorDict({}, [4])
        sub_tensordict = TensorDict({}, [4, 5])
        sub_sub_tensordict = TensorDict({"c": tensor}, [4, 5, 6])
        sub_sub_tensordict2 = TensorDict({"c": tensor2}, [4, 5, 6])
        sub_tensordict["b"] = sub_sub_tensordict
        tensordict["a"] = sub_tensordict
        assert tensordict["a", "b"] is sub_sub_tensordict
        tensordict["a", "b"] = sub_sub_tensordict2
        assert tensordict["a", "b"] is sub_sub_tensordict2
        assert (tensordict["a", "b", "c"] == 1).all()

        # check the same with set method
        sub_tensordict.set("b", sub_sub_tensordict)
        tensordict.set("a", sub_tensordict)
        assert tensordict["a", "b"] is sub_sub_tensordict

        tensordict.set(("a", "b"), sub_sub_tensordict2)
        assert tensordict["a", "b"] is sub_sub_tensordict2
        assert (tensordict["a", "b", "c"] == 1).all()

    def test_shared_inheritance(self):
        td = TensorDict({"a": torch.randn(3, 4)}, [3, 4])
        td.share_memory_()

        td0, *_ = td.unbind(1)
        assert td0.is_shared()

        td0, *_ = td.split(1, 0)
        assert td0.is_shared()

        td0 = td.exclude("a")
        assert td0.is_shared()

        td0 = td.select("a")
        assert td0.is_shared()

        td.unlock_()
        td0 = td.rename_key_("a", "a.a")
        assert not td0.is_shared()
        td.share_memory_()

        td0 = td.unflatten_keys(".")
        assert td0.is_shared()

        td0 = td.flatten_keys(".")
        assert td0.is_shared()

        td0 = td.view(-1)
        assert not td0.is_shared()

        td0 = td.permute(1, 0)
        assert not td0.is_shared()

        td0 = td.unsqueeze(0)
        assert not td0.is_shared()

        td0 = td0.squeeze(0)
        assert not td0.is_shared()

    def test_split_with_empty_tensordict(self):
        td = TensorDict({}, [10])

        tds = td.split(4, 0)
        assert len(tds) == 3
        assert tds[0].shape == torch.Size([4])
        assert tds[1].shape == torch.Size([4])
        assert tds[2].shape == torch.Size([2])

        tds = td.split([1, 9], 0)

        assert len(tds) == 2
        assert tds[0].shape == torch.Size([1])
        assert tds[1].shape == torch.Size([9])

        td = TensorDict({}, [10, 10, 3])

        tds = td.split(4, 1)
        assert len(tds) == 3
        assert tds[0].shape == torch.Size([10, 4, 3])
        assert tds[1].shape == torch.Size([10, 4, 3])
        assert tds[2].shape == torch.Size([10, 2, 3])

        tds = td.split([1, 9], 1)
        assert len(tds) == 2
        assert tds[0].shape == torch.Size([10, 1, 3])
        assert tds[1].shape == torch.Size([10, 9, 3])

    def test_split_with_invalid_arguments(self):
        td = TensorDict({"a": torch.zeros(2, 1)}, [])
        # Test empty batch size
        with self.assertRaisesRegex(
            RuntimeError,
            expected_regex="TensorDict with empty batch size is not splittable",
        ):
            td.split(1, 0)

        td = TensorDict({}, [3, 2])

        # Test invalid split_size input
        with self.assertRaisesRegex(
            TypeError,
            expected_regex=re.escape(
                "split(): argument 'split_size' must be int or list of ints"
            ),
        ):
            td.split("1", 0)
        with self.assertRaisesRegex(
            TypeError,
            expected_regex=re.escape(
                "split(): argument 'split_size' must be int or list of ints"
            ),
        ):
            td.split(["1", 2], 0)

        # Test invalid split_size sum
        with self.assertRaisesRegex(
            RuntimeError,
            expected_regex=re.escape(
                "Split method expects split_size to sum exactly to 3 (tensor's size at dimension 0), but got split_size=[]"
            ),
        ):
            td.split([], 0)

        with self.assertRaisesRegex(
            RuntimeError,
            expected_regex=re.escape(
                "Split method expects split_size to sum exactly to 3 (tensor's size at dimension 0), but got split_size=[1, 1]"
            ),
        ):
            td.split([1, 1], 0)

        # Test invalid dimension input
        with self.assertRaisesRegex(
            IndexError, expected_regex=re.escape("Dimension out of range")
        ):
            td.split(1, 2)
        with self.assertRaisesRegex(
            IndexError, expected_regex=re.escape("Dimension out of range")
        ):
            td.split(1, -3)

    def test_split_with_negative_dim(self):
        td = TensorDict(
            {"a": torch.zeros(5, 4, 2, 1), "b": torch.zeros(5, 4, 1)}, [5, 4]
        )

        tds = td.split([1, 3], -1)
        assert len(tds) == 2
        assert tds[0].shape == torch.Size([5, 1])
        assert tds[0]["a"].shape == torch.Size([5, 1, 2, 1])
        assert tds[0]["b"].shape == torch.Size([5, 1, 1])
        assert tds[1].shape == torch.Size([5, 3])
        assert tds[1]["a"].shape == torch.Size([5, 3, 2, 1])
        assert tds[1]["b"].shape == torch.Size([5, 3, 1])

    @parametrize("device", get_available_devices())
    def test_subtensordict_construction(self, device):
        torch.manual_seed(1)
        td = TensorDict({}, batch_size=(4, 5))
        val1 = torch.randn(4, 5, 1, device=device)
        val2 = torch.randn(4, 5, 6, dtype=torch.double, device=device)
        val1_copy = val1.clone()
        val2_copy = val2.clone()
        td.set("key1", val1)
        td.set("key2", val2)
        std1 = td._get_sub_tensordict(2)
        std2 = std1._get_sub_tensordict(2)
        idx = (2, 2)
        std_control = td._get_sub_tensordict(idx)
        assert (std_control.get("key1") == std2.get("key1")).all()
        assert (std_control.get("key2") == std2.get("key2")).all()

        # write values
        with self.assertRaisesRegex(
            RuntimeError,
            expected_regex=re.escape(
                "Calling `_SubTensorDict.set(key, value, inplace=False)` is prohibited for existing tensors. Consider calling _SubTensorDict.set_(...) or cloning your tensordict first."
            ),
        ):
            std_control.set("key1", torch.randn(1, device=device))
        with self.assertRaisesRegex(
            RuntimeError,
            expected_regex=re.escape(
                "Calling `_SubTensorDict.set(key, value, inplace=False)` is prohibited for existing tensors. Consider calling _SubTensorDict.set_(...) or cloning your tensordict first."
            ),
        ):
            std_control.set("key2", torch.randn(6, device=device, dtype=torch.double))

        subval1 = torch.randn(1, device=device)
        subval2 = torch.randn(6, device=device, dtype=torch.double)
        std_control.set_("key1", subval1)
        std_control.set_("key2", subval2)
        assert (val1_copy[idx] != subval1).all()
        assert (td.get("key1")[idx] == subval1).all()
        assert (td.get("key1")[1, 1] == val1_copy[1, 1]).all()

        assert (val2_copy[idx] != subval2).all()
        assert (td.get("key2")[idx] == subval2).all()
        assert (td.get("key2")[1, 1] == val2_copy[1, 1]).all()

        assert (std_control.get("key1") == std2.get("key1")).all()
        assert (std_control.get("key2") == std2.get("key2")).all()

        assert std_control.get_parent_tensordict() is td
        assert (
            std_control.get_parent_tensordict()
            is std2.get_parent_tensordict().get_parent_tensordict()
        )

    @parametrize("device", get_available_devices())
    def test_tensordict_device(self, device):
        tensordict = TensorDict({"a": torch.randn(3, 4)}, [])
        assert tensordict.device is None

        tensordict = TensorDict({"a": torch.randn(3, 4, device=device)}, [])
        assert tensordict["a"].device == device
        assert tensordict.device is None

        tensordict = TensorDict(
            {
                "a": torch.randn(3, 4, device=device),
                "b": torch.randn(3, 4),
                "c": torch.randn(3, 4, device="cpu"),
            },
            [],
            device=device,
        )
        assert tensordict.device == device
        assert tensordict["a"].device == device
        assert tensordict["b"].device == device
        assert tensordict["c"].device == device

        tensordict = TensorDict({}, [], device=device)
        tensordict["a"] = torch.randn(3, 4)
        tensordict["b"] = torch.randn(3, 4, device="cpu")
        assert tensordict["a"].device == device
        assert tensordict["b"].device == device

        tensordict = TensorDict({"a": torch.randn(3, 4)}, [])
        tensordict = tensordict.to(device)
        assert tensordict.device == device
        assert tensordict["a"].device == device

    def test_tensordict_error_messages(self):
        if not torch.cuda.device_count():
            self.skipTest("No cuda device detected")
        sub1 = TensorDict({"a": torch.randn(2, 3)}, [2])
        sub2 = TensorDict({"a": torch.randn(2, 3, device="cuda")}, [2])
        td1 = TensorDict({"sub": sub1}, [2])
        td2 = TensorDict({"sub": sub2}, [2])

        with self.assertRaisesRegex(
            RuntimeError,
            expected_regex='tensors on different devices at key "sub" / "a"',
        ):
            torch.cat([td1, td2], 0)

    @parametrize("device", get_available_devices())
    def test_tensordict_indexing(self, device):
        torch.manual_seed(1)
        td = TensorDict({}, batch_size=(4, 5))
        td.set("key1", torch.randn(4, 5, 1, device=device))
        td.set("key2", torch.randn(4, 5, 6, device=device, dtype=torch.double))

        td_select = td[2, 2]
        td_select._check_batch_size()

        td_select = td[2, :2]
        td_select._check_batch_size()

        td_select = td[None, :2]
        td_select._check_batch_size()

        td_reconstruct = torch.stack(list(td), 0)
        assert (
            td_reconstruct == td
        ).all(), f"td and td_reconstruct differ, got {td} and {td_reconstruct}"

        superlist = [torch.stack(list(_td), 0) for _td in td]
        td_reconstruct = torch.stack(superlist, 0)
        assert (
            td_reconstruct == td
        ).all(), f"td and td_reconstruct differ, got {td == td_reconstruct}"

        x = torch.randn(4, 5, device=device)
        td = TensorDict(
            source={"key1": torch.zeros(3, 4, 5, device=device)},
            batch_size=[3, 4],
        )
        td[0].set_("key1", x)
        torch.testing.assert_close(td.get("key1")[0], x)
        torch.testing.assert_close(td.get("key1")[0], td[0].get("key1"))

        y = torch.randn(3, 5, device=device)
        td[:, 0].set_("key1", y)
        torch.testing.assert_close(td.get("key1")[:, 0], y)
        torch.testing.assert_close(td.get("key1")[:, 0], td[:, 0].get("key1"))

    def test_tensordict_prealloc_nested(self):
        N = 3
        B = 5
        T = 4
        buffer = TensorDict({}, batch_size=[B, N])

        td_0 = TensorDict(
            {
                "env.time": torch.rand(N, 1),
                "agent.obs": TensorDict(
                    {  # assuming 3 agents in a multi-agent setting
                        "image": torch.rand(N, T, 64),
                        "state": torch.rand(N, T, 3, 32, 32),
                    },
                    batch_size=[N, T],
                ),
            },
            batch_size=[N],
        )

        td_1 = td_0.clone()
        buffer[0] = td_0
        buffer[1] = td_1
        assert (
            repr(buffer)
            == """TensorDict(
    fields={
        agent.obs: TensorDict(
            fields={
                image: Tensor(shape=torch.Size([5, 3, 4, 64]), device=cpu, dtype=torch.float32, is_shared=False),
                state: Tensor(shape=torch.Size([5, 3, 4, 3, 32, 32]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([5, 3, 4]),
            device=None,
            is_shared=False),
        env.time: Tensor(shape=torch.Size([5, 3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
    batch_size=torch.Size([5, 3]),
    device=None,
    is_shared=False)"""
        ), repr(buffer)
        assert buffer.batch_size == torch.Size([B, N])
        assert buffer["agent.obs"].batch_size == torch.Size([B, N, T])

    @parametrize("device", get_available_devices())
    def test_tensordict_set(self, device):
        torch.manual_seed(1)
        td = TensorDict({}, batch_size=(4, 5), device=device)
        td.set("key1", torch.randn(4, 5))
        assert td.device == torch.device(device)
        # by default inplace:
        with self.assertRaises(RuntimeError):
            td.set("key1", torch.randn(5, 5, device=device))

        # robust to dtype casting
        td.set_("key1", torch.ones(4, 5, device=device, dtype=torch.double))
        assert (td.get("key1") == 1).all()

        # robust to device casting
        td.set("key_device", torch.ones(4, 5, device="cpu", dtype=torch.double))
        assert td.get("key_device").device == torch.device(device)

        with self.assertRaisesRegex(
            KeyError,
            expected_regex=re.escape(
                'key "smartypants" not found in TensorDict with keys'
            ),
        ):
            td.set_("smartypants", torch.ones(4, 5, device="cpu", dtype=torch.double))
        # test set_at_
        td.set("key2", torch.randn(4, 5, 6, device=device))
        x = torch.randn(6, device=device)
        td.set_at_("key2", x, (2, 2))
        assert (td.get("key2")[2, 2] == x).all()

        # test set_at_ with dtype casting
        x = torch.randn(6, dtype=torch.double, device=device)
        td.set_at_("key2", x, (2, 2))  # robust to dtype casting
        torch.testing.assert_close(td.get("key2")[2, 2], x.to(torch.float))

        td.set(
            "key1", torch.zeros(4, 5, dtype=torch.double, device=device), inplace=True
        )
        assert (td.get("key1") == 0).all()
        td.set(
            "key1",
            torch.randn(4, 5, 1, 2, dtype=torch.double, device=device),
            inplace=False,
        )
        assert td["key1"].shape == td._tensordict["key1"].shape

    def test_unbind_batchsize(self):
        td = TensorDict({"a": TensorDict({"b": torch.zeros(2, 3)}, [2, 3])}, [2])
        td["a"].batch_size
        tds = td.unbind(0)
        assert tds[0].batch_size == torch.Size([])
        assert tds[0]["a"].batch_size == torch.Size([3])

    @parametrize("device", get_available_devices())
    def test_unbind_td(self, device):
        torch.manual_seed(1)
        d = {
            "key1": torch.randn(4, 5, 6, device=device),
            "key2": torch.randn(4, 5, 10, device=device),
        }
        td = TensorDict(batch_size=(4, 5), source=d)
        td_unbind = torch.unbind(td, dim=1)
        assert (
            td_unbind[0].batch_size == td[:, 0].batch_size
        ), f"got {td_unbind[0].batch_size} and {td[:, 0].batch_size}"

    def test_update_nested_dict(self):
        t = TensorDict({"a": {"d": [[[0]] * 3] * 2}}, [2, 3])
        assert ("a", "d") in t.keys(include_nested=True)
        t.update({"a": {"b": [[[1]] * 3] * 2}})
        assert ("a", "d") in t.keys(include_nested=True)
        assert ("a", "b") in t.keys(include_nested=True)
        assert t["a", "b"].shape == torch.Size([2, 3, 1])
        t.update({"a": {"d": [[[1]] * 3] * 2}})

    @parametrize("nested", [False, True])
    @parametrize("td_type", TD_TYPES)
    def test_add_batch_dim_cache(self, td_type, nested):
        td = getattr(self, td_type)
        if nested:
            td = TensorDict({"parent": td}, td.batch_size)
        from torch import vmap

        fun = vmap(lambda x: x)
        fun(td)

        if td_type == "td_params":
            with self.assertRaisesRegex(
                RuntimeError, expected_regex="leaf Variable that requires grad"
            ):
                td.zero_()
            return

        td.zero_()
        # this value should be cached
        std = fun(td)
        for value in std.values(True, True):
            assert (value == 0).all()

    @parametrize("inplace", [False, True])
    @parametrize("td_type", TD_TYPES)
    def test_apply(self, td_type, inplace):
        td = getattr(self, td_type)
        td_c = td.to_tensordict()
        if inplace and td_type == "td_params":
            with self.assertRaisesRegex(ValueError, expected_regex="Failed to update"):
                td.apply(lambda x: x + 1, inplace=inplace)
            return
        td_1 = td.apply(lambda x: x + 1, inplace=inplace)
        if inplace:
            for key in td.keys(True, True):
                assert (td_c[key] + 1 == td[key]).all()
                assert (td_1[key] == td[key]).all()
        else:
            for key in td.keys(True, True):
                assert (td_c[key] + 1 != td[key]).any()
                assert (td_1[key] == td[key] + 1).all()

    @parametrize("inplace", [False, True])
    @parametrize("td_type", TD_TYPES)
    def test_apply_other(self, td_type, inplace):
        td = getattr(self, td_type)
        td_c = td.to_tensordict()
        if inplace and td_type == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_1 = td_set.apply(lambda x, y: x + y, td_c, inplace=inplace)
        if inplace:
            for key in td.keys(True, True):
                assert (td_c[key] * 2 == td[key]).all()
                assert (td_1[key] == td[key]).all()
        else:
            for key in td.keys(True, True):
                assert (td_c[key] * 2 != td[key]).any()
                assert (td_1[key] == td[key] * 2).all()

    @parametrize("td_type", TD_TYPES)
    def test_assert(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        with self.assertRaisesRegex(
            RuntimeError,
            expected_regex=re.escape(
                "Converting a tensordict to boolean value is not permitted"
            ),
        ):
            assert td

    @parametrize("td_type", TD_TYPES)
    def test_broadcast(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        sub_td = td[:, :2].to_tensordict()
        sub_td.zero_()
        sub_dict = sub_td.to_dict()
        if td_type == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set[:, :2] = sub_dict
        assert (td[:, :2] == 0).all()

    @parametrize("td_type", TD_TYPES)
    @parametrize("op", ["flatten", "unflatten"])
    def test_cache(self, td_type, op):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        try:
            td.lock_()
        except Exception:
            return
        if op == "keys_root":
            a = list(td.keys())
            b = list(td.keys())
            assert a == b
        elif op == "keys_nested":
            a = list(td.keys(True))
            b = list(td.keys(True))
            assert a == b
        elif op == "values":
            a = list(td.values(True))
            b = list(td.values(True))
            assert all((_a == _b).all() for _a, _b in zip(a, b))
        elif op == "items":
            keys_a, values_a = zip(*td.items(True))
            keys_b, values_b = zip(*td.items(True))
            assert all((_a == _b).all() for _a, _b in zip(values_a, values_b))
            assert keys_a == keys_b
        elif op == "flatten":
            a = td.flatten_keys()
            b = td.flatten_keys()
            assert a is b
        elif op == "unflatten":
            a = td.unflatten_keys()
            b = td.unflatten_keys()
            assert a is b

        if td_type != "td_params":
            assert len(td._cache)
        td.unlock_()
        assert td._cache is None
        for val in td.values(True):
            if is_tensor_collection(val):
                assert td._cache is None

    @parametrize("device_cast", get_available_devices())
    @parametrize("td_type", TD_TYPES)
    def test_cast_device(self, td_type, device_cast):
        if not torch.cuda.device_count():
            self.skipTest("No cuda device detected")
        torch.manual_seed(1)
        td = getattr(self, td_type)
        td_device = td.to(device_cast)

        for item in td_device.values():
            assert item.device == device_cast
        for item in td_device.clone().values():
            assert item.device == device_cast

        assert td_device.device == device_cast, (
            f"td_device first tensor device is " f"{next(td_device.items())[1].device}"
        )
        assert td_device.clone().device == device_cast
        if device_cast != td.device:
            assert td_device is not td
        assert td_device.to(device_cast) is td_device
        device = td.device
        if device is not None:
            assert td.to(device) is td
            assert_allclose_td(td, td_device.to(device))

    @parametrize("td_type", TD_TYPES)
    def test_cast_to(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        td_device = td.to("cpu:1")
        assert td_device.device == torch.device("cpu:1")
        td_dtype = td.to(torch.int)
        assert all(t.dtype == torch.int for t in td_dtype.values(True, True))
        del td_dtype
        # device (str), dtype
        td_dtype_device = td.to("cpu:1", torch.int)
        assert all(t.dtype == torch.int for t in td_dtype_device.values(True, True))
        assert td_dtype_device.device == torch.device("cpu:1")
        del td_dtype_device
        # device, dtype
        td_dtype_device = td.to(torch.device("cpu:1"), torch.int)
        assert all(t.dtype == torch.int for t in td_dtype_device.values(True, True))
        assert td_dtype_device.device == torch.device("cpu:1")
        del td_dtype_device
        # example tensor
        td_dtype_device = td.to(torch.randn(3, dtype=torch.half, device="cpu:1"))
        assert all(t.dtype == torch.half for t in td_dtype_device.values(True, True))
        # tensor on cpu:1 is actually on cpu. This is still meaningful for tensordicts on cuda.
        assert td_dtype_device.device == torch.device("cpu")
        del td_dtype_device
        # example td
        td_dtype_device = td.to(
            other=TensorDict(
                {"a": torch.randn(3, dtype=torch.half, device="cpu:1")},
                [],
                device="cpu:1",
            )
        )
        assert all(t.dtype == torch.half for t in td_dtype_device.values(True, True))
        assert td_dtype_device.device == torch.device("cpu:1")
        del td_dtype_device
        # example td, many dtypes
        td_nodtype_device = td.to(
            other=TensorDict(
                {
                    "a": torch.randn(3, dtype=torch.half, device="cpu:1"),
                    "b": torch.randint(10, ()),
                },
                [],
                device="cpu:1",
            )
        )
        assert all(t.dtype != torch.half for t in td_nodtype_device.values(True, True))
        assert td_nodtype_device.device == torch.device("cpu:1")
        del td_nodtype_device
        # batch-size: check errors (or not)
        td_dtype_device = td.to(
            torch.device("cpu:1"), torch.int, batch_size=torch.Size([])
        )
        assert all(t.dtype == torch.int for t in td_dtype_device.values(True, True))
        assert td_dtype_device.device == torch.device("cpu:1")
        assert td_dtype_device.batch_size == torch.Size([])
        del td_dtype_device
        td_batchsize = td.to(batch_size=torch.Size([]))
        assert td_batchsize.batch_size == torch.Size([])
        del td_batchsize

    @parametrize("td_type", TD_TYPES)
    def test_casts(self, td_type):
        td = getattr(self, td_type)
        tdfloat = td.float()
        assert all(value.dtype is torch.float for value in tdfloat.values(True, True))
        tddouble = td.double()
        assert all(value.dtype is torch.double for value in tddouble.values(True, True))
        tdbfloat16 = td.bfloat16()
        assert all(
            value.dtype is torch.bfloat16 for value in tdbfloat16.values(True, True)
        )
        tdhalf = td.half()
        assert all(value.dtype is torch.half for value in tdhalf.values(True, True))
        tdint = td.int()
        assert all(value.dtype is torch.int for value in tdint.values(True, True))
        tdint = td.type(torch.int)
        assert all(value.dtype is torch.int for value in tdint.values(True, True))

    @parametrize("dim", [0, 1])
    @parametrize("chunks", [1, 2])
    @parametrize("td_type", TD_TYPES)
    def test_chunk(self, td_type, dim, chunks):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        if len(td.shape) - 1 < dim:
            self.skipTest(f"no dim {dim} in td")

        chunks = min(td.shape[dim], chunks)
        td_chunks = td.chunk(chunks, dim)
        assert len(td_chunks) == chunks
        assert sum([_td.shape[dim] for _td in td_chunks]) == td.shape[dim]
        assert (torch.cat(td_chunks, dim) == td).all()

    @parametrize("td_type", TD_TYPES)
    def test_clone_td(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        clone = torch.clone(td)
        assert (clone == td).all()
        assert td.batch_size == clone.batch_size
        assert type(td.clone(recurse=False)) is type(td)
        assert td.clone(recurse=False).get("a") is td.get("a")

    @parametrize("td_type", TD_TYPES)
    def test_cpu_cuda(self, td_type):
        if torch.cuda.device_count() == 0:
            self.skipTest("no cuda device detected.")
        torch.manual_seed(1)
        td = getattr(self, td_type)
        td_device = td.cuda()
        td_back = td_device.cpu()
        assert td_device.device == torch.device("cuda")
        assert td_back.device == torch.device("cpu")

    @parametrize("td_type", TD_TYPES)
    def test_create_nested(self, td_type):
        td = getattr(self, td_type)
        with td.unlock_():
            td.create_nested("root")
            assert td.get("root").shape == td.shape
            assert is_tensor_collection(td.get("root"))
            td.create_nested(("some", "nested", "key"))
            assert td.get(("some", "nested", "key")).shape == td.shape
            assert is_tensor_collection(td.get(("some", "nested", "key")))
        with td.lock_(), self.assertRaises(RuntimeError):
            td.create_nested("root")

    @parametrize("td_type", TD_TYPES)
    def test_default_nested(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        default_val = torch.randn(())
        timbers = td.get(("shiver", "my", "timbers"), default_val)
        assert timbers == default_val

    @parametrize("td_type", TD_TYPES)
    def test_delitem(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        if td.is_locked:
            with self.assertRaisesRegex(
                RuntimeError,
                expected_regex=re.escape(
                    "Cannot modify locked TensorDict. For in-place modification, consider using the `set_()` method and make sure the key is present."
                ),
            ):
                del td["a"]
            return
        else:
            del td["a"]
        assert "a" not in td.keys()

    @parametrize("td_type", TD_TYPES)
    def test_empty_like(self, td_type):
        td = getattr(self, td_type)
        td_empty = torch.empty_like(td)
        if td_type == "td_params":
            with self.assertRaisesRegex(ValueError, expected_regex="Failed to update"):
                td.apply_(lambda x: x + 1.0)
            return

        td.apply_(lambda x: x + 1.0)
        assert type(td) is type(td_empty)
        assert all(val.any() for val in (td != td_empty).values(True, True))

    @parametrize("td_type", TD_TYPES)
    def test_enter_exit(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        is_locked = td.is_locked
        with td.lock_() as other:
            assert other is td
            assert td.is_locked
            with td.unlock_() as other:
                assert other is td
                assert not td.is_locked
            assert td.is_locked
        assert td.is_locked is is_locked

    @parametrize("td_type", TD_TYPES)
    def test_entry_type(self, td_type):
        td = getattr(self, td_type)
        for key in td.keys(include_nested=True):
            assert type(td.get(key)) is td.entry_class(key)

    @parametrize("td_type", TD_TYPES)
    def test_equal(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        assert (td == td.to_tensordict()).all()
        td0 = td.to_tensordict().zero_()
        assert (td != td0).any()

    @parametrize("td_type", TD_TYPES)
    def test_equal_dict(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        assert (td == td.to_dict()).all()
        td0 = td.to_tensordict().zero_().to_dict()
        assert (td != td0).any()

    @parametrize("td_type", TD_TYPES)
    def test_equal_float(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        if td_type == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set.zero_()
        assert (td == 0.0).all()
        td0 = td.clone()
        if td_type == "td_params":
            td_set = td0.data
        else:
            td_set = td0
        td_set.zero_()
        assert (td0 != 1.0).all()

    @parametrize("td_type", TD_TYPES)
    def test_equal_int(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        if td_type == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set.zero_()
        assert (td == 0).all()
        td0 = td.to_tensordict().zero_()
        assert (td0 != 1).all()

    @parametrize("td_type", TD_TYPES)
    def test_equal_other(self, td_type):
        td = getattr(self, td_type)
        assert not td == "z"
        assert td != "z"

    @parametrize("td_type", TD_TYPES)
    def test_equal_tensor(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        if td_type == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set.zero_()
        device = td.device
        assert (td == torch.zeros([], dtype=torch.int, device=device)).all()
        td0 = td.to_tensordict().zero_()
        assert (td0 != torch.ones([], dtype=torch.int, device=device)).all()

    @parametrize("td_type", TD_TYPES)
    def test_exclude(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        td2 = td.exclude("a")
        assert td2 is not td
        assert (
            len(list(td2.keys())) == len(list(td.keys())) - 1 and "a" not in td2.keys()
        )
        assert (
            len(list(td2.clone().keys())) == len(list(td.keys())) - 1
            and "a" not in td2.clone().keys()
        )

        with td.unlock_():
            td2 = td.exclude("a", inplace=True)
        assert td2 is td

    @parametrize("nested", [True, False])
    @parametrize("td_type", TD_TYPES)
    def test_exclude_missing(self, td_type, nested):
        td = getattr(self, td_type)
        if nested:
            td2 = td.exclude("this key is missing", ("this one too",))
        else:
            td2 = td.exclude(
                "this key is missing",
            )
        assert (td == td2).all()

    @parametrize("nested", [True, False])
    @parametrize("td_type", TD_TYPES)
    def test_exclude_nested(self, td_type, nested):
        td = getattr(self, td_type)
        td.unlock_()  # make sure that the td is not locked
        td["newnested", "first"] = torch.randn(td.shape)
        if nested:
            td2 = td.exclude("a", ("newnested", "first"))
            assert "a" in td.keys(), list(td.keys())
            assert "a" not in td2.keys()
            assert ("newnested", "first") in td.keys(True), list(td.keys(True))
            assert ("newnested", "first") not in td2.keys(True)
        else:
            td2 = td.exclude(
                "a",
            )
            assert "a" in td.keys()
            assert "a" not in td2.keys()
        if td_type not in ("td_params",):
            assert type(td2) is type(td)

    @parametrize("td_type", TD_TYPES)
    def test_expand(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        batch_size = td.batch_size
        expected_size = torch.Size([3, *batch_size])

        new_td = td.expand(3, *batch_size)
        assert new_td.batch_size == expected_size
        assert all((_new_td == td).all() for _new_td in new_td)

        new_td_torch_size = td.expand(expected_size)
        assert new_td_torch_size.batch_size == expected_size
        assert all((_new_td == td).all() for _new_td in new_td_torch_size)

        new_td_iterable = td.expand([3, *batch_size])
        assert new_td_iterable.batch_size == expected_size
        assert all((_new_td == td).all() for _new_td in new_td_iterable)

    @parametrize("td_type", TD_TYPES)
    def test_fill_(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        if td_type == "td_params":
            td_set = td.data
        else:
            td_set = td
        new_td = td_set.fill_("a", 0.1)
        assert (td.get("a") == 0.1).all()
        assert new_td is td_set

    @parametrize("inplace", [True, False])
    @parametrize("separator", [",", "-"])
    @parametrize("td_type", TD_TYPES)
    def test_flatten_keys(self, td_type, inplace, separator):
        td = getattr(self, td_type)
        locked = td.is_locked
        td.unlock_()
        nested_nested_tensordict = TensorDict(
            {
                "a": torch.zeros(*td.shape, 2, 3),
            },
            [*td.shape, 2],
        )
        nested_tensordict = TensorDict(
            {
                "a": torch.zeros(*td.shape, 2),
                "nested_nested_tensordict": nested_nested_tensordict,
            },
            td.shape,
        )
        td["nested_tensordict"] = nested_tensordict
        if locked:
            td.lock_()

        if inplace and locked:
            with self.assertRaisesRegex(
                RuntimeError, expected_regex="Cannot modify locked TensorDict"
            ):
                td_flatten = td.flatten_keys(inplace=inplace, separator=separator)
            return
        else:
            td_flatten = td.flatten_keys(inplace=inplace, separator=separator)
        for value in td_flatten.values():
            assert not isinstance(value, TensorDictBase)
        assert (
            separator.join(["nested_tensordict", "nested_nested_tensordict", "a"])
            in td_flatten.keys()
        )
        if inplace:
            assert td_flatten is td
        else:
            assert td_flatten is not td

    @parametrize("td_type", TD_TYPES)
    def test_flatten_unflatten(self, td_type):
        td = getattr(self, td_type)
        shape = td.shape[:3]
        td_flat = td.flatten(0, 2)
        td_unflat = td_flat.unflatten(0, shape)
        assert (td.to_tensordict() == td_unflat).all()
        assert td.batch_size == td_unflat.batch_size

    @parametrize("td_type", TD_TYPES)
    def test_flatten_unflatten_bis(self, td_type):
        td = getattr(self, td_type)
        shape = td.shape[1:4]
        td_flat = td.flatten(1, 3)
        td_unflat = td_flat.unflatten(1, shape)
        assert (td.to_tensordict() == td_unflat).all()
        assert td.batch_size == td_unflat.batch_size

    @parametrize("td_type", TD_TYPES)
    def test_from_empty(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        new_td = TensorDict({}, batch_size=td.batch_size, device=td.device)
        for key, item in td.items():
            new_td.set(key, item)
        assert_allclose_td(td, new_td)
        assert td.device == new_td.device
        assert td.shape == new_td.shape

    @parametrize(
        "actual_index,expected_index",
        [
            (..., (slice(None),) * TD_BATCH_SIZE),
            ((..., 0), (slice(None),) * (TD_BATCH_SIZE - 1) + (0,)),
            ((0, ...), (0,) + (slice(None),) * (TD_BATCH_SIZE - 1)),
            ((0, ..., 0), (0,) + (slice(None),) * (TD_BATCH_SIZE - 2) + (0,)),
        ],
    )
    @parametrize("td_type", TD_TYPES)
    def test_getitem_ellipsis(self, td_type, actual_index, expected_index):
        torch.manual_seed(1)

        td = getattr(self, td_type)

        actual_td = td[actual_index]
        expected_td = td[expected_index]
        other_expected_td = td.to_tensordict()[expected_index]
        assert expected_td.shape == _getitem_batch_size(
            td.batch_size, convert_ellipsis_to_idx(actual_index, td.batch_size)
        )
        assert other_expected_td.shape == actual_td.shape
        assert_allclose_td(actual_td, other_expected_td)
        assert_allclose_td(actual_td, expected_td)

    @parametrize("td_type", TD_TYPES)
    def test_getitem_nestedtuple(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        assert isinstance(td[(("a",))], torch.Tensor)
        assert isinstance(td.get((("a",))), torch.Tensor)

    @parametrize("td_type", TD_TYPES)
    def test_getitem_range(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        assert_allclose_td(td[range(2)], td[[0, 1]])
        assert td[range(1), range(1)].shape == td[[0], [0]].shape
        assert_allclose_td(td[range(1), range(1)], td[[0], [0]])

        assert_allclose_td(td[:, range(2)], td[:, [0, 1]])
        assert_allclose_td(td[..., range(1)], td[..., [0]])

    @parametrize("td_type", TD_TYPES)
    def test_getitem_string(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        assert isinstance(td["a"], torch.Tensor)

    @parametrize(
        "idx",
        [
            (..., None),
            (None, ...),
            (None,),
            None,
            (slice(None), None),
            (0, None),
            (None, slice(None), slice(None)),
            (None, ..., None),
            (None, 1, ..., None),
            (1, ..., None),
            (..., None, 0),
            ([1], ..., None),
        ],
    )
    @parametrize("td_type", TD_TYPES)
    def test_index_none(self, td_type, idx):
        td = getattr(self, td_type)
        tdnone = td[idx]
        tensor = torch.zeros(td.shape)
        assert tdnone.shape == tensor[idx].shape, idx
        # Fixed by 451
        # if td_type == "td_h5":
        #     with self.assertRaisesRegex(TypeError, expected_regex="can't process None"):
        #         assert (tdnone.to_tensordict() == td.to_tensordict()[idx]).all()
        #     return
        assert (tdnone.to_tensordict() == td.to_tensordict()[idx]).all()

    @parametrize("td_type", TD_TYPES)
    def test_indexed_properties(self, td_type):
        td = getattr(self, td_type)
        td_index = td[0]
        assert td_index.is_memmap() is td.is_memmap()
        assert td_index.is_shared() is td.is_shared()
        assert td_index.device == td.device

    @parametrize("td_type", TD_TYPES)
    def test_inferred_view_size(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        for i in range(len(td.shape)):
            # replacing every index one at a time
            # with -1, to test that td.view(..., -1, ...)
            # always returns the original tensordict
            new_shape = [
                dim_size if dim_idx != i else -1
                for dim_idx, dim_size in enumerate(td.shape)
            ]
            td_view = td.view(*new_shape)
            for v1, v2 in zip(td_view.values(True), td.values(True)):
                assert v1.shape == v2.shape
            td_view = td.view(-1).view(*new_shape)
            for v1, v2 in zip(td_view.values(True), td.values(True)):
                assert v1.shape == v2.shape

    @parametrize("td_type", TD_TYPES)
    def test_items_values_keys(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        td.unlock_()
        keys = list(td.keys())
        values = list(td.values())
        items = list(td.items())

        # Test td.items()
        constructed_td1 = TensorDict({}, batch_size=td.shape)
        for key, value in items:
            constructed_td1.set(key, value)

        assert (td == constructed_td1).all()

        # Test td.keys() and td.values()
        # items = [key, value] should be verified
        assert len(values) == len(items)
        assert len(keys) == len(items)
        constructed_td2 = TensorDict({}, batch_size=td.shape)
        for key, value in list(zip(td.keys(), td.values())):
            constructed_td2.set(key, value)

        assert (td == constructed_td2).all()

        # Test that keys is sorted
        assert all(keys[i] <= keys[i + 1] for i in range(len(keys) - 1))

        # Add new element to tensor
        a = td.get("a")
        td.set("x", torch.randn_like(a))
        keys = list(td.keys())
        values = list(td.values())
        items = list(td.items())

        # Test that keys is still sorted after adding the element
        assert all(keys[i] <= keys[i + 1] for i in range(len(keys) - 1))

        # Test td.items()
        # after adding the new element
        constructed_td1 = TensorDict({}, batch_size=td.shape)
        for key, value in items:
            constructed_td1.set(key, value)

        assert (td == constructed_td1).all()

        # Test td.keys() and td.values()
        # items = [key, value] should be verified
        # even after adding the new element
        assert len(values) == len(items)
        assert len(keys) == len(items)

        constructed_td2 = TensorDict({}, batch_size=td.shape)
        for key, value in list(zip(td.keys(), td.values())):
            constructed_td2.set(key, value)

        assert (td == constructed_td2).all()

    @parametrize("td_type", TD_TYPES)
    def test_lock(self, td_type):
        td = getattr(self, td_type)
        is_locked = td.is_locked
        if td_type != "td_params":
            # for td_params, things are a bit special
            # we want to avoid people messing up with inner tensordicts so
            # we lock the content of the TensorDictParams but not the TensorDictParams
            # itself. Hence the root attribute and the nodes attributes won't match.
            for item in td.values():
                if isinstance(item, TensorDictBase):
                    assert item.is_locked == is_locked
        td.is_locked = not is_locked
        assert td.is_locked != is_locked
        if td_type != "td_params":
            for value in td.values():
                if isinstance(value, TensorDictBase):
                    assert value.is_locked != is_locked
        td.lock_()
        assert td.is_locked
        if td_type != "td_params":
            for value in td.values():
                if isinstance(value, TensorDictBase):
                    assert value.is_locked
        td.unlock_()
        assert not td.is_locked
        if td_type != "td_params":
            for value in td.values():
                if isinstance(value, TensorDictBase):
                    assert not value.is_locked

    @parametrize("td_type", TD_TYPES)
    def test_lock_change_names(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        try:
            td.names = [str(i) for i in range(td.ndim)]
            td.lock_()
        except Exception:
            return
        # cache values
        list(td.values(True))
        td.names = [str(-i) for i in range(td.ndim)]
        for val in td.values(True):
            if not is_tensor_collection(val):
                continue
            assert val.names[: td.ndim] == [str(-i) for i in range(td.ndim)]

    @parametrize("td_type", TD_TYPES)
    def test_lock_nested(self, td_type):
        td = getattr(self, td_type)
        td.unlock_()
        td.set(("some", "nested"), torch.zeros(td.shape))
        td.lock_()
        some = td.get("some")
        assert some.is_locked
        with self.assertRaises(RuntimeError):
            some.unlock_()
        del td
        some.unlock_()

    @parametrize("td_type", TD_TYPES)
    def test_lock_write(self, td_type):
        td = getattr(self, td_type)
        td.lock_()
        td_clone = td.clone()
        assert not td_clone.is_locked
        td_clone = td.to_tensordict()
        assert not td_clone.is_locked
        assert td.is_locked
        with self.assertRaises(RuntimeError):
            # cannot empty a locked tensordict
            td = td.select(inplace=True)
        for key, item in td_clone.items(True):
            with self.assertRaisesRegex(
                RuntimeError, expected_regex="Cannot modify locked TensorDict"
            ):
                td.set(key, item)
        td.unlock_()
        for key, item in td_clone.items(True):
            td.set(key, item)
        td.lock_()
        for key, item in td_clone.items(True):
            with self.assertRaisesRegex(
                RuntimeError, expected_regex="Cannot modify locked TensorDict"
            ):
                td.set(key, item)
            if td_type == "td_params":
                td_set = td.data
            else:
                td_set = td
            td_set.set_(key, item)

    @parametrize("td_type", TD_TYPES)
    def test_masked_fill(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        device = td.device
        mask = torch.zeros(td.shape, dtype=torch.bool, device=device).bernoulli_()
        new_td = td.masked_fill(mask, -10.0)
        assert new_td is not td
        for item in new_td.values():
            assert (item[mask] == -10).all()

    @parametrize("td_type", TD_TYPES)
    def test_masked_fill_(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        device = td.device
        mask = torch.zeros(td.shape, dtype=torch.bool, device=device).bernoulli_()
        if td_type == "td_params":
            td_set = td.data
        else:
            td_set = td
        new_td = td_set.masked_fill_(mask, -10.0)
        assert new_td is td_set
        for item in td.values():
            assert (item[mask] == -10).all(), item[mask]

    @parametrize("td_type", TD_TYPES)
    def test_masking(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        device = td.device
        while True:
            mask = torch.zeros(
                td.batch_size, dtype=torch.bool, device=device
            ).bernoulli_(0.8)
            if not mask.all() and mask.any():
                break
        td_masked = td[mask]
        td_masked2 = torch.masked_select(td, mask)
        assert_allclose_td(td_masked, td_masked2)
        assert td_masked.batch_size[0] == mask.sum()
        assert td_masked.batch_dims == 1

    @parametrize("td_type", TD_TYPES)
    def test_masking_set(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        device = td.device
        mask = torch.zeros(td.batch_size, dtype=torch.bool, device=device).bernoulli_(
            0.8
        )
        n = mask.sum()
        d = td.ndimension()
        pseudo_td = td.apply(
            lambda item: torch.zeros(
                (n, *item.shape[d:]), dtype=item.dtype, device=device
            ),
            batch_size=[n, *td.batch_size[d:]],
        )
        # the inner batch-size of nested tensordicts must match
        for key, value in pseudo_td.items(True):
            if is_tensor_collection(value):
                value.batch_size = [n, *td.get(key).batch_size[d:]]

        if td_type == "td_params":
            td_set = td.data
        else:
            td_set = td

        td_set[mask] = pseudo_td
        for item in td.values():
            assert (item[mask] == 0).all()

    @parametrize("td_type", TD_TYPES)
    def test_memmap_(self, td_type):
        td = getattr(self, td_type)
        if td_type in ("td_params",):
            with self.assertRaisesRegex(
                RuntimeError,
                expected_regex="Cannot build a memmap TensorDict in-place",
            ):
                td.memmap_()
        else:
            td.memmap_()
            assert td.is_memmap()

    @parametrize("copy_existing", [False, True])
    @parametrize("td_type", TD_TYPES)
    def test_memmap_existing(self, td_type, copy_existing):
        if td_type in ("td_params",):
            self.skipTest("memmap_ incompatibility is checked elsewhere")
        with TemporaryDirectoryName() as tmp_path:
            td = getattr(self, td_type).memmap_(
                prefix=os.path.join(tmp_path, "tensordict")
            )
            td2 = getattr(self, td_type).memmap_()

            if copy_existing:
                td3 = td.memmap_(
                    prefix=os.path.join(tmp_path, "tensordict2"), copy_existing=True
                )
                assert (td == td3).all()
            else:
                with self.assertRaisesRegex(
                    RuntimeError,
                    expected_regex=re.escape(
                        "A filename was provided but the tensor already has a file associated"
                    ),
                ):
                    # calling memmap_ with prefix that is different to contents gives error
                    td.memmap_(prefix=os.path.join(tmp_path, "tensordict2"))

                # calling memmap_ without prefix means no-op, regardless of whether contents
                # were saved in temporary or designated location (td vs. td2 resp.)
                td3 = td.memmap_()
                td4 = td2.memmap_()
                assert all(
                    td3[key] is value
                    for key, value in td.items(include_nested=True, leaves_only=True)
                )
                assert all(
                    td4[key] is value
                    for key, value in td2.items(include_nested=True, leaves_only=True)
                )

    @parametrize("td_type", TD_TYPES)
    def test_memmap_like(self, td_type):
        td = getattr(self, td_type)
        tdmemmap = td.memmap_like()
        assert tdmemmap is not td
        for key in td.keys(True):
            assert td[key] is not tdmemmap[key]
        assert (tdmemmap == 0).all()

    @parametrize("td_type", TD_TYPES)
    def test_memmap_prefix(self, td_type):
        with TemporaryDirectoryName() as tmp_path:
            td = getattr(self, td_type)
            if td_type in ("td_params",):
                with self.assertRaisesRegex(
                    RuntimeError,
                    expected_regex="Cannot build a memmap TensorDict in-place",
                ):
                    td.memmap_(os.path.join(tmp_path, "tensordict"))
                return
            else:
                td.memmap_(os.path.join(tmp_path, "tensordict"))

            assert os.path.exists(os.path.join(tmp_path, "tensordict", "meta.json"))
            with open(os.path.join(tmp_path, "tensordict", "meta.json"), "r") as file:
                metadata = json.load(file)
            assert td.batch_size == torch.Size(metadata["shape"])
            device = (
                torch.device(metadata["device"])
                if metadata["device"] != "None"
                else None
            )
            assert td.device == device, (td.device, device)

            td2 = td.__class__.load_memmap(os.path.join(tmp_path, "tensordict"))
            assert (td == td2).all()

    @parametrize("td_type", TD_TYPES)
    def test_nested_dict_init(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        device = td.device

        td.unlock_()

        # Create TensorDict and dict equivalent values, and populate each with according nested value
        td_clone = td.clone(recurse=True)
        td_dict = td.to_dict()
        nested_dict_value = {"e": torch.randn(4, 3, 2, 1, 10)}
        nested_tensordict_value = TensorDict(
            nested_dict_value, batch_size=td.batch_size, device=device
        )
        td_dict["d"] = nested_dict_value
        td_clone["d"] = nested_tensordict_value

        # Re-init new TensorDict from dict, and check if they're equal
        td_dict_init = TensorDict(td_dict, batch_size=td.batch_size, device=device)

        assert (td_clone == td_dict_init).all()

    @parametrize("td_type", TD_TYPES)
    def test_nested_td(self, td_type):
        td = getattr(self, td_type)
        device = td.device
        td.unlock_()
        tdin = TensorDict({"inner": torch.randn(td.shape)}, td.shape, device=device)
        td.set("inner_td", tdin)
        assert (td["inner_td"] == tdin).all()

    @parametrize("td_type", TD_TYPES)
    def test_nested_td_emptyshape(self, td_type):
        td = getattr(self, td_type)
        device = td.device
        td.unlock_()
        tdin = TensorDict({"inner": torch.randn(*td.shape, 1)}, [], device=device)
        td["inner_td"] = tdin
        tdin.batch_size = td.batch_size
        assert (td["inner_td"] == tdin).all()

    @parametrize("td_type", TD_TYPES)
    def test_nested_td_index(self, td_type):
        td = getattr(self, td_type)
        device = td.device
        td.unlock_()

        sub_td = TensorDict({}, [*td.shape, 2], device=device)
        a = torch.zeros([*td.shape, 2, 2], device=device)
        sub_sub_td = TensorDict({"a": a}, [*td.shape, 2, 2], device=device)
        sub_td.set("sub_sub_td", sub_sub_td)
        td.set("sub_td", sub_td)
        assert (td["sub_td", "sub_sub_td", "a"] == 0).all()
        assert (
            td["sub_td"]["sub_sub_td"]["a"] == td["sub_td", "sub_sub_td", "a"]
        ).all()

        a = torch.ones_like(a)
        other_sub_sub_td = TensorDict({"a": a}, [*td.shape, 2, 2])
        td["sub_td", "sub_sub_td"] = other_sub_sub_td
        assert (td["sub_td", "sub_sub_td", "a"] == 1).all()
        assert (
            td["sub_td"]["sub_sub_td"]["a"] == td["sub_td", "sub_sub_td", "a"]
        ).all()

        b = torch.ones_like(a)
        other_sub_sub_td = TensorDict({"b": b}, [*td.shape, 2, 2])

        td["sub_td", "sub_sub_td"] = other_sub_sub_td
        assert (td["sub_td", "sub_sub_td", "b"] == 1).all()
        assert (
            td["sub_td"]["sub_sub_td"]["b"] == td["sub_td", "sub_sub_td", "b"]
        ).all()

    @parametrize("td_type", TD_TYPES)
    def test_pad(self, td_type):
        td = getattr(self, td_type)
        paddings = [
            [0, 1, 0, 2],
            [1, 0, 0, 2],
            [1, 0, 2, 1],
        ]

        for pad_size in paddings:
            padded_td = pad(td, pad_size)
            padded_td._check_batch_size()
            amount_expanded = [0] * (len(pad_size) // 2)
            for i in range(0, len(pad_size), 2):
                amount_expanded[i // 2] = pad_size[i] + pad_size[i + 1]

            for key in padded_td.keys():
                expected_dims = tuple(
                    sum(p)
                    for p in zip(
                        td[key].shape,
                        amount_expanded
                        + [0] * (len(td[key].shape) - len(amount_expanded)),
                    )
                )
                assert padded_td[key].shape == expected_dims

        with self.assertRaises(RuntimeError):
            pad(td, [0] * 100)

        with self.assertRaises(RuntimeError):
            pad(td, [0])

    @parametrize("td_type", TD_TYPES)
    def test_permute_applied_twice(self, td_type):
        torch.manual_seed(0)
        tensordict = getattr(self, td_type)
        for _ in range(10):
            p = torch.randperm(4)
            inv_p = p.argsort()
            other_p = inv_p
            while (other_p == inv_p).all():
                other_p = torch.randperm(4)
            other_p = tuple(other_p.tolist())
            p = tuple(p.tolist())
            inv_p = tuple(inv_p.tolist())
            circled = tensordict.permute(*p).permute(*inv_p)
            for v1, v2 in zip(
                circled.values(include_nested=True),
                tensordict.values(include_nested=True),
            ):
                assert v1.shape == v2.shape
            circled = tensordict.permute(*p).permute(*other_p)
            for v1, v2 in zip(
                circled.values(include_nested=True),
                tensordict.values(include_nested=True),
            ):
                if v1.shape != v2.shape:
                    break
            else:
                raise AssertionError
            circled = tensordict.permute(p).permute(inv_p)
            for v1, v2 in zip(
                circled.values(include_nested=True),
                tensordict.values(include_nested=True),
            ):
                assert v1.shape == v2.shape
            circled = tensordict.permute(p).permute(other_p)
            for v1, v2 in zip(
                circled.values(include_nested=True),
                tensordict.values(include_nested=True),
            ):
                if v1.shape != v2.shape:
                    break
            else:
                raise AssertionError

    @parametrize("device_cast", [0, "cuda:0", torch.device("cuda:0")])
    @parametrize("td_type", TD_TYPES)
    def test_pin_memory(self, td_type, device_cast):
        if torch.cuda.device_count() == 0:
            self.skipTest("No cuda device detected")
        torch.manual_seed(1)
        td = getattr(self, td_type)
        device = td.device
        td.unlock_()
        if device.type == "cuda":
            with self.assertRaisesRegex(RuntimeError, expected_regex="cannot pin"):
                td.pin_memory()
            return
        td.pin_memory()
        td_device = td.to(device_cast)
        _device_cast = torch.device(device_cast)
        assert td_device.device == _device_cast
        assert td_device.clone().device == _device_cast
        if device != _device_cast:
            assert td_device is not td
        for item in td_device.values():
            assert item.device == _device_cast
        for item in td_device.clone().values():
            assert item.device == _device_cast
        # assert type(td_device) is type(td)
        assert_allclose_td(td, td_device.to(device))

    @parametrize("td_type", TD_TYPES)
    def test_pop(self, td_type):
        td = getattr(self, td_type)
        device = td.device
        assert "a" in td.keys()
        a = td["a"].clone()
        with td.unlock_():
            out = td.pop("a")
            assert (out == a).all()
            assert "a" not in td.keys()

            assert "b" in td.keys()
            b = td["b"].clone()
            default = torch.zeros_like(b).to(device)
            assert (default != b).all()
            out = td.pop("b", default)

            assert torch.ne(out, default).all()
            assert (out == b).all()

            assert "z" not in td.keys()
            out = td.pop("z", default)
            assert (out == default).all()

            with self.assertRaisesRegex(
                KeyError,
                expected_regex=re.escape(r"You are trying to pop key"),
            ):
                td.pop("z")

    @parametrize("call_del", [True, False])
    @parametrize("td_type", TD_TYPES)
    def test_remove(self, td_type, call_del):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        with td.unlock_():
            if call_del:
                del td["a"]
            else:
                td = td.del_("a")
        assert td is not None
        assert "a" not in td.keys()
        if td_type in ("sub_td", "sub_td2"):
            return
        td.lock_()
        with self.assertRaisesRegex(RuntimeError, expected_regex="locked"):
            del td["b"]

    @parametrize("td_type", TD_TYPES)
    def test_rename_key_nested(self, td_type) -> None:
        torch.manual_seed(1)
        td = getattr(self, td_type)
        td.unlock_()
        td["nested", "conflict"] = torch.zeros(td.shape)
        with self.assertRaisesRegex(
            KeyError, expected_regex="already present in TensorDict"
        ):
            td.rename_key_(("nested", "conflict"), "b", safe=True)
        td["nested", "first"] = torch.zeros(td.shape)
        td.rename_key_(("nested", "first"), "second")
        assert (td["second"] == 0).all()
        assert ("nested", "first") not in td.keys(True)
        td.rename_key_("second", ("nested", "back"))
        assert (td[("nested", "back")] == 0).all()
        assert "second" not in td.keys()

    @parametrize("td_type", TD_TYPES)
    def test_repr(self, td_type):
        td = getattr(self, td_type)
        _ = str(td)

    @parametrize("td_type", TD_TYPES)
    def test_reshape(self, td_type):
        td = getattr(self, td_type)
        td_reshape = td.reshape(td.shape)
        assert td_reshape.shape.numel() == td.shape.numel()
        assert td_reshape.shape == td.shape
        td_reshape = td.reshape(*td.shape)
        assert td_reshape.shape.numel() == td.shape.numel()
        assert td_reshape.shape == td.shape
        td_reshape = td.reshape(size=td.shape)
        assert td_reshape.shape.numel() == td.shape.numel()
        assert td_reshape.shape == td.shape
        td_reshape = td.reshape(-1)
        assert td_reshape.shape.numel() == td.shape.numel()
        assert td_reshape.shape == torch.Size([td.shape.numel()])
        td_reshape = td.reshape((-1,))
        assert td_reshape.shape.numel() == td.shape.numel()
        assert td_reshape.shape == torch.Size([td.shape.numel()])
        td_reshape = td.reshape(size=(-1,))
        assert td_reshape.shape.numel() == td.shape.numel()
        assert td_reshape.shape == torch.Size([td.shape.numel()])

    @parametrize("strict", [True, False])
    @parametrize("inplace", [True, False])
    @parametrize("td_type", TD_TYPES)
    def test_select(self, td_type, strict, inplace):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        keys = ["a"]
        if td_type == "td_h5":
            with self.assertRaisesRegex(
                NotImplementedError, expected_regex="Cannot call select"
            ):
                td.select(*keys, strict=strict, inplace=inplace)
            return

        if td_type in ("nested_stacked_td", "nested_td"):
            keys += [("my_nested_td", "inner")]

        if inplace and td.is_locked:
            with self.assertRaisesRegex(
                RuntimeError,
                expected_regex=re.escape(
                    "Cannot modify locked TensorDict. For in-place modification, consider using the `set_()` method and make sure the key is present."
                ),
            ):
                td.select(*keys, strict=strict, inplace=inplace)
            return
        td2 = td.select(*keys, strict=strict, inplace=inplace)
        if inplace:
            assert td2 is td
        else:
            assert td2 is not td
        if td_type == "saved_td":
            assert (len(list(td2.keys())) == len(keys)) and ("a" in td2.keys())
            assert (len(list(td2.clone().keys())) == len(keys)) and (
                "a" in td2.clone().keys()
            )
        else:
            assert (len(list(td2.keys(True, True))) == len(keys)) and (
                "a" in td2.keys()
            )
            assert (len(list(td2.clone().keys(True, True))) == len(keys)) and (
                "a" in td2.clone().keys()
            )

    @parametrize("strict", [True, False])
    @parametrize("td_type", TD_TYPES)
    def test_select_exception(self, td_type, strict):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        if td_type == "td_h5":
            with self.assertRaisesRegex(
                NotImplementedError, expected_regex="Cannot call select"
            ):
                _ = td.select("tada", strict=strict)
            return

        if strict:
            with self.assertRaises(KeyError):
                _ = td.select("tada", strict=strict)
        else:
            td2 = td.select("tada", strict=strict)
            assert td2 is not td
            assert len(list(td2.keys())) == 0

    @parametrize("td_type", TD_TYPES)
    def test_set_nested_batch_size(self, td_type):
        td = getattr(self, td_type)
        td.unlock_()
        batch_size = torch.Size([*td.batch_size, 3])
        td.set("some_other_td", TensorDict({}, batch_size))
        assert td["some_other_td"].batch_size == batch_size

    @parametrize("td_type", TD_TYPES)
    def test_set_nontensor(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        td.unlock_()
        r = torch.randn_like(td.get("a"))
        td.set("numpy", r.cpu().numpy())
        torch.testing.assert_close(td.get("numpy"), r)

    @parametrize("td_type", TD_TYPES)
    def test_set_requires_grad(self, td_type):
        td = getattr(self, td_type)
        if td_type in ("td_params",):
            td.apply(lambda x: x.requires_grad_(False))
        td.unlock_()
        assert not td.get("a").requires_grad
        if td_type in ("td_h5",):
            with self.assertRaisesRegex(
                RuntimeError,
                expected_regex="Cannot set a tensor that has requires_grad=True",
            ):
                td.set("a", torch.randn_like(td.get("a")).requires_grad_())
            return
        if td_type in ("sub_td", "sub_td2"):
            td.set_("a", torch.randn_like(td.get("a")).requires_grad_())
        else:
            td.set("a", torch.randn_like(td.get("a")).requires_grad_())

        assert td.get("a").requires_grad

    @parametrize("td_type", TD_TYPES)
    def test_set_unexisting(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        if td.is_locked:
            with self.assertRaisesRegex(
                RuntimeError,
                expected_regex="Cannot modify locked TensorDict. For in-place modification",
            ):
                td.set("z", torch.ones_like(td.get("a")))
        else:
            td.set("z", torch.ones_like(td.get("a")))
            assert (td.get("z") == 1).all()

    @parametrize("td_type", TD_TYPES)
    def test_setdefault_existing_key(self, td_type):
        td = getattr(self, td_type)
        td.unlock_()
        expected = td.get("a")
        inserted = td.setdefault("a", torch.ones_like(td.get("b")))
        assert (inserted == expected).all()

    @parametrize("td_type", TD_TYPES)
    def test_setdefault_missing_key(self, td_type):
        td = getattr(self, td_type)
        td.unlock_()
        expected = torch.ones_like(td.get("a"))
        inserted = td.setdefault("z", expected)
        assert (inserted == expected).all()

    @parametrize("td_type", TD_TYPES)
    def test_setdefault_nested(self, td_type):
        td = getattr(self, td_type)
        device = td.device
        td.unlock_()

        tensor = torch.randn(4, 3, 2, 1, 5, device=device)
        tensor2 = torch.ones(4, 3, 2, 1, 5, device=device)
        sub_sub_tensordict = TensorDict({"c": tensor}, [4, 3, 2, 1], device=device)
        sub_tensordict = TensorDict(
            {"b": sub_sub_tensordict}, [4, 3, 2, 1], device=device
        )
        if td_type == "td_h5":
            del td["a"]
        if td_type == "sub_td":
            td = td._source.set(
                "a", sub_tensordict.expand(2, *sub_tensordict.shape)
            ).get_sub_tensordict(1)
        elif td_type == "sub_td2":
            td = td._source.set(
                "a",
                sub_tensordict.expand(2, *sub_tensordict.shape).permute(1, 0, 2, 3, 4),
            ).get_sub_tensordict((slice(None), 1))
        else:
            td.set("a", sub_tensordict)

        # if key exists we return the existing value
        torch.testing.assert_close(td.setdefault(("a", "b", "c"), tensor2), tensor)

        if not td_type == "stacked_td":
            torch.testing.assert_close(td.setdefault(("a", "b", "d"), tensor2), tensor2)
            torch.testing.assert_close(td.get(("a", "b", "d")), tensor2)

    @parametrize(
        "idx", [slice(1), torch.tensor([0]), torch.tensor([0, 1]), range(1), range(2)]
    )
    @parametrize("td_type", TD_TYPES)
    def test_setitem(self, td_type, idx):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        if isinstance(idx, torch.Tensor) and idx.numel() > 1 and td.shape[0] == 1:
            self.skipTest("cannot index tensor with desired index")

        td_clone = td[idx].to_tensordict().zero_()
        if td_type == "td_params":
            td.data[idx] = td_clone
        else:
            td[idx] = td_clone
        assert (td[idx].get("a") == 0).all()

        td_clone = torch.cat([td_clone, td_clone], 0)
        with self.assertRaisesRegex(
            RuntimeError,
            expected_regex=r"differs from the source batch size|batch dimension mismatch|Cannot broadcast the tensordict",
        ):
            td[idx] = td_clone

    @parametrize("actual_index", [..., (..., 0), (0, ...), (0, ..., 0)])
    @parametrize("td_type", TD_TYPES)
    def test_setitem_ellipsis(self, td_type, actual_index):
        torch.manual_seed(1)
        td = getattr(self, td_type)

        idx = actual_index
        td_clone = td.clone()
        actual_td = td_clone[idx].clone()
        if td_type in ("td_params",):
            td_set = actual_td.apply(lambda x: x.data)
        else:
            td_set = actual_td
        td_set.zero_()

        for key in actual_td.keys():
            assert (actual_td.get(key) == 0).all()

        if td_type in ("td_params",):
            td_set = td_clone.data
        else:
            td_set = td_clone

        td_set[idx] = actual_td
        for key in td_clone.keys():
            assert (td_clone[idx].get(key) == 0).all()

    @parametrize("td_type", TD_TYPES)
    def test_setitem_nested_dict_value(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        device = td.device

        # Create equivalent TensorDict and dict nested values for setitem
        nested_dict_value = {"e": torch.randn(4, 3, 2, 1, 10)}
        nested_tensordict_value = TensorDict(
            nested_dict_value, batch_size=td.batch_size, device=device
        )
        td_clone1 = td.clone(recurse=True)
        td_clone2 = td.clone(recurse=True)

        td_clone1["d"] = nested_dict_value
        td_clone2["d"] = nested_tensordict_value
        assert (td_clone1 == td_clone2).all()

    @parametrize("td_type", TD_TYPES)
    def test_setitem_nestedtuple(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        if td.is_locked:
            td.unlock_()
        td[" a ", (("little", "story")), "about", ("myself",)] = torch.zeros(td.shape)
        assert (td[" a ", "little", "story", "about", "myself"] == 0).all()

    @parametrize("td_type", TD_TYPES)
    def test_setitem_slice(self, td_type):
        td = getattr(self, td_type)
        if td_type == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set[:] = td.clone()
        td_set[:1] = td[:1].clone().zero_()
        assert (td[:1] == 0).all()
        td = getattr(self, td_type)
        if td_type == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set[:1] = td[:1].to_tensordict().zero_()
        assert (td[:1] == 0).all()

        # with broadcast
        td = getattr(self, td_type)
        if td_type == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set[:1] = td[0].clone().zero_()
        assert (td[:1] == 0).all()
        td = getattr(self, td_type)
        if td_type == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set[:1] = td[0].to_tensordict().zero_()
        assert (td[:1] == 0).all()

        td = getattr(self, td_type)
        if td_type == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set[:1, 0] = td[0, 0].clone().zero_()
        assert (td[:1, 0] == 0).all()
        td = getattr(self, td_type)
        if td_type == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set[:1, 0] = td[0, 0].to_tensordict().zero_()
        assert (td[:1, 0] == 0).all()

        td = getattr(self, td_type)
        if td_type == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set[:1, :, 0] = td[0, :, 0].clone().zero_()
        assert (td[:1, :, 0] == 0).all()
        td = getattr(self, td_type)
        if td_type == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set[:1, :, 0] = td[0, :, 0].to_tensordict().zero_()
        assert (td[:1, :, 0] == 0).all()

    @parametrize("td_type", TD_TYPES)
    def test_setitem_string(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        td.unlock_()
        td["d"] = torch.randn(4, 3, 2, 1, 5)
        assert "d" in td.keys()

    @parametrize("td_type", TD_TYPES)
    def test_shape(self, td_type):
        td = getattr(self, td_type)
        assert td.shape == td.batch_size

    @parametrize("td_type", TD_TYPES)
    def test_sorted_keys(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        sorted_keys = td.sorted_keys
        i = -1
        for i, (key1, key2) in enumerate(zip(sorted_keys, td.keys())):  # noqa: B007
            assert key1 == key2
        assert i == len(td.keys()) - 1
        if td.is_locked:
            assert td._cache.get("sorted_keys", None) is not None
            td.unlock_()
            assert td._cache is None
        elif td_type not in ("sub_td", "sub_td2"):  # we cannot lock sub tensordicts
            target = td
            assert target._cache is None
            td.lock_()
            _ = td.sorted_keys
            assert target._cache.get("sorted_keys", None) is not None
            td.unlock_()
            assert target._cache is None

    @parametrize("performer", ["torch", "tensordict"])
    @parametrize("td_type", TD_TYPES)
    def test_split(self, td_type, performer):
        td = getattr(self, td_type)

        for dim in range(td.batch_dims):
            rep, remainder = divmod(td.shape[dim], 2)
            length = rep + remainder

            # split_sizes to be [2, 2, ..., 2, 1] or [2, 2, ..., 2]
            split_sizes = [2] * rep + [1] * remainder
            for test_split_size in (2, split_sizes):
                if performer == "torch":
                    tds = torch.split(td, test_split_size, dim)
                elif performer == "tensordict":
                    tds = td.split(test_split_size, dim)
                assert len(tds) == length

                for idx, split_td in enumerate(tds):
                    expected_split_dim_size = 1 if idx == rep else 2
                    expected_batch_size = [
                        expected_split_dim_size if dim_idx == dim else dim_size
                        for (dim_idx, dim_size) in enumerate(td.batch_size)
                    ]

                    # Test each split_td has the expected batch_size
                    assert split_td.batch_size == torch.Size(expected_batch_size)

                    if td_type == "nested_td":
                        assert isinstance(split_td["my_nested_td"], TensorDict)
                        assert isinstance(
                            split_td["my_nested_td"]["inner"], torch.Tensor
                        )

                    # Test each tensor (or nested_td) in split_td has the expected shape
                    for key, item in split_td.items():
                        expected_shape = [
                            expected_split_dim_size if dim_idx == dim else dim_size
                            for (dim_idx, dim_size) in enumerate(td[key].shape)
                        ]
                        assert item.shape == torch.Size(expected_shape)

                        if key == "my_nested_td":
                            expected_inner_tensor_size = [
                                expected_split_dim_size if dim_idx == dim else dim_size
                                for (dim_idx, dim_size) in enumerate(
                                    td[key]["inner"].shape
                                )
                            ]
                            assert item["inner"].shape == torch.Size(
                                expected_inner_tensor_size
                            )

    @parametrize("td_type", TD_TYPES)
    def test_squeeze(self, td_type, squeeze_dim=-1):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        td.unlock_()  # make sure that the td is not locked
        td_squeeze = torch.squeeze(td, dim=-1)
        tensor_squeeze_dim = td.batch_dims + squeeze_dim
        tensor = torch.ones_like(td.get("a").squeeze(tensor_squeeze_dim))
        if td_type in ("sub_td", "sub_td2"):
            td_squeeze.set_("a", tensor)
        else:
            td_squeeze.set("a", tensor)
        assert td.batch_size[squeeze_dim] == 1
        assert (td_squeeze.get("a") == tensor).all()
        assert (td_squeeze.get("a") == 1).all()

    @parametrize("td_type", TD_TYPES)
    def test_squeeze_with_none(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        td_squeeze = torch.squeeze(td, dim=None)
        tensor = torch.ones_like(td.get("a").squeeze())
        if td_type == "td_params":
            with self.assertRaisesRegex(ValueError, expected_regex="Failed to update"):
                td_squeeze.set_("a", tensor)
            return
        td_squeeze.set_("a", tensor)
        assert (td_squeeze.get("a") == tensor).all()
        if td_type == "unsqueezed_td":
            assert td_squeeze._source is td
        assert (td_squeeze.get("a") == 1).all()
        assert (td.get("a") == 1).all()

    @parametrize("td_type", TD_TYPES)
    def test_stack_onto(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        td0 = td.clone()
        if td_type in ("td_params",):
            td0.data.apply_(lambda x: x.zero_())
        else:
            td0.apply_(lambda x: x.zero_())
        td1 = td.clone()
        if td_type in ("td_params",):
            td1.data.apply_(lambda x: x.zero_() + 1)
        else:
            td1.apply_(lambda x: x.zero_() + 1)

        td_out = td.unsqueeze(1).expand(td.shape[0], 2, *td.shape[1:]).clone()
        td_stack = torch.stack([td0, td1], 1)
        if td_type == "td_params":
            with self.assertRaisesRegex(
                RuntimeError, expected_regex="out.batch_size and stacked"
            ):
                torch.stack([td0, td1], 0, out=td_out)
            return
        data_ptr_set_before = {val.data_ptr() for val in decompose(td_out)}
        torch.stack([td0, td1], 1, out=td_out)
        data_ptr_set_after = {val.data_ptr() for val in decompose(td_out)}
        assert data_ptr_set_before == data_ptr_set_after
        assert (td_stack == td_out).all()

    @parametrize("td_type", TD_TYPES)
    def test_stack_subclasses_on_td(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        td = td.expand(3, *td.batch_size).clone().zero_()
        tds_list = [getattr(self, td_type) for _ in range(3)]
        if td_type == "td_params":
            with self.assertRaisesRegex(
                RuntimeError, expected_regex="arguments don't support automatic"
            ):
                torch.stack(tds_list, 0, out=td)
            return
        data_ptr_set_before = {val.data_ptr() for val in decompose(td)}
        stacked_td = torch.stack(tds_list, 0, out=td)
        data_ptr_set_after = {val.data_ptr() for val in decompose(td)}
        assert data_ptr_set_before == data_ptr_set_after
        assert stacked_td.batch_size == td.batch_size
        for key in ("a", "b", "c"):
            assert (stacked_td[key] == td[key]).all()

    @parametrize("td_type", TD_TYPES)
    def test_stack_tds_on_subclass(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        device = td.device
        tds_count = td.batch_size[0]
        tds_batch_size = td.batch_size[1:]
        tds_list = [
            TensorDict(
                source={
                    "a": torch.ones(*tds_batch_size, 5),
                    "b": torch.ones(*tds_batch_size, 10),
                    "c": torch.ones(*tds_batch_size, 3, dtype=torch.long),
                },
                batch_size=tds_batch_size,
                device=device,
            )
            for _ in range(tds_count)
        ]
        if td_type in ("sub_td", "sub_td2"):
            with self.assertRaisesRegex(
                IndexError, expected_regex="storages of the indexed tensors"
            ):
                torch.stack(tds_list, 0, out=td)
            return
        if td_type == "td_params":
            with self.assertRaisesRegex(
                RuntimeError, expected_regex="arguments don't support automatic"
            ):
                torch.stack(tds_list, 0, out=td)
            return
        data_ptr_set_before = {val.data_ptr() for val in decompose(td)}

        stacked_td = torch.stack(tds_list, 0, out=td)
        data_ptr_set_after = {val.data_ptr() for val in decompose(td)}
        assert data_ptr_set_before == data_ptr_set_after
        assert stacked_td.batch_size == td.batch_size
        assert stacked_td is td
        for key in ("a", "b", "c"):
            assert (stacked_td[key] == 1).all()

    @parametrize("td_type", TD_TYPES)
    def test_state_dict(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        sd = td.state_dict()
        td_zero = td.clone().detach().zero_()
        td_zero.load_state_dict(sd)
        assert_allclose_td(td, td_zero)

    @parametrize("td_type", TD_TYPES)
    def test_state_dict_assign(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        sd = td.state_dict()
        td_zero = td.clone().detach().zero_()
        shallow_copy = td_zero.clone(False)
        td_zero.load_state_dict(sd, assign=True)
        assert (shallow_copy == 0).all()
        assert_allclose_td(td, td_zero)

    @parametrize("td_type", TD_TYPES)
    def test_state_dict_strict(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        sd = td.state_dict()
        td_zero = td.clone().detach().zero_()
        del sd["a"]
        td_zero.load_state_dict(sd, strict=False)
        with self.assertRaises(RuntimeError):
            td_zero.load_state_dict(sd, strict=True)

    @parametrize("td_type", TD_TYPES)
    def test_tensordict_set(self, td_type):
        torch.manual_seed(1)
        np.random.seed(1)
        td = getattr(self, td_type)
        td.unlock_()

        # test set
        val1 = np.ones(shape=(4, 3, 2, 1, 10))
        td.set("key1", val1)
        assert (td.get("key1") == 1).all()
        with self.assertRaises(RuntimeError):
            td.set("key1", np.ones(shape=(5, 10)))

        # test set_
        val2 = np.zeros(shape=(4, 3, 2, 1, 10))
        td.set_("key1", val2)
        assert (td.get("key1") == 0).all()
        if td_type not in ("stacked_td", "nested_stacked_td"):
            err_msg = r"key.*smartypants.*not found in "
        elif td_type in ("td_h5",):
            err_msg = "Unable to open object"
        else:
            err_msg = "setting a value in-place on a stack of TensorDict"

        with self.assertRaisesRegex(KeyError, expected_regex=err_msg):
            td.set_("smartypants", np.ones(shape=(4, 3, 2, 1, 5)))

        # test set_at_
        td.set("key2", np.random.randn(4, 3, 2, 1, 5))
        x = np.ones(shape=(2, 1, 5)) * 42
        td.set_at_("key2", x, (2, 2))
        assert (td.get("key2")[2, 2] == 42).all()

    @parametrize("td_type", TD_TYPES)
    def test_tensordict_set_dict_value(self, td_type):
        torch.manual_seed(1)
        np.random.seed(1)
        td = getattr(self, td_type)
        td.unlock_()

        # test set
        val1 = {"subkey1": torch.ones(4, 3, 2, 1, 10)}
        td.set("key1", val1)
        assert (td.get("key1").get("subkey1") == 1).all()
        with self.assertRaises(RuntimeError):
            td.set("key1", torch.ones(5, 10))

        # test set_
        val2 = {"subkey1": torch.zeros(4, 3, 2, 1, 10)}
        if td_type in ("td_params",):
            td.data.set_("key1", val2)
        else:
            td.set_("key1", val2)
        assert (td.get("key1").get("subkey1") == 0).all()

        if td_type not in ("stacked_td", "nested_stacked_td"):
            err_msg = r"key.*smartypants.*not found in "
        elif td_type in ("td_h5",):
            err_msg = "Unable to open object"
        else:
            err_msg = "setting a value in-place on a stack of TensorDict"

        with self.assertRaisesRegex(KeyError, expected_regex=err_msg):
            td.set_("smartypants", np.ones(shape=(4, 3, 2, 1, 5)))

    @parametrize("td_type", TD_TYPES)
    def test_to_dict_nested(self, td_type):
        def recursive_checker(cur_dict):
            for _, value in cur_dict.items():
                if isinstance(value, TensorDict):
                    return False
                elif isinstance(value, dict) and not recursive_checker(value):
                    return False
            return True

        td = getattr(self, td_type)
        device = td.device
        td.unlock_()

        # Create nested TensorDict
        nested_tensordict_value = TensorDict(
            {"e": torch.randn(4, 3, 2, 1, 10)}, batch_size=td.batch_size, device=device
        )
        td["d"] = nested_tensordict_value

        # Convert into dictionary and recursively check if the values are TensorDicts
        td_dict = td.to_dict()
        assert recursive_checker(td_dict)

    @parametrize("td_type", TD_TYPES)
    def test_to_tensordict(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        td2 = td.to_tensordict()
        assert (td2 == td).all()

    @parametrize("td_type", TD_TYPES)
    def test_transpose(self, td_type):
        td = getattr(self, td_type)
        tdt = td.transpose(0, 1)
        assert tdt.shape == torch.Size([td.shape[1], td.shape[0], *td.shape[2:]])
        for key, value in tdt.items(True):
            assert value.shape == torch.Size(
                [td.get(key).shape[1], td.get(key).shape[0], *td.get(key).shape[2:]]
            )
        tdt = td.transpose(-1, -2)
        for key, value in tdt.items(True):
            assert value.shape == td.get(key).transpose(2, 3).shape
        with td.unlock_():
            tdt.set(("some", "transposed", "tensor"), torch.zeros(tdt.shape))
        with self.assertRaisesRegex(
            ValueError, expected_regex="The provided dimensions are incompatible"
        ):
            td.transpose(-5, -6)
        with self.assertRaisesRegex(
            ValueError, expected_regex="The provided dimensions are incompatible"
        ):
            tdt.transpose(-5, -6)

    @parametrize("td_type", TD_TYPES)
    @parametrize("dim", range(4))
    def test_unbind(self, td_type, dim):
        if td_type not in ["sub_td", "idx_td", "td_reset_bs"]:
            torch.manual_seed(1)
            td = getattr(self, td_type)
            td_unbind = torch.unbind(td, dim=dim)
            assert (td == torch.stack(td_unbind, dim)).all()
            idx = (slice(None),) * dim + (0,)
            assert (td[idx] == td_unbind[0]).all()

    @parametrize("td_type", TD_TYPES)
    @parametrize("inplace", [True, False])
    @parametrize("separator", [",", "-"])
    def test_unflatten_keys(self, td_type, inplace, separator):
        td = getattr(self, td_type)
        locked = td.is_locked
        td.unlock_()
        nested_nested_tensordict = TensorDict(
            {
                "a": torch.zeros(*td.shape, 2, 3),
            },
            [*td.shape, 2],
        )
        nested_tensordict = TensorDict(
            {
                "a": torch.zeros(*td.shape, 2),
                "nested_nested_tensordict": nested_nested_tensordict,
            },
            td.shape,
        )
        td["nested_tensordict"] = nested_tensordict

        if inplace and locked:
            td_flatten = td.flatten_keys(inplace=inplace, separator=separator)
            td_flatten.lock_()
            with self.assertRaisesRegex(
                RuntimeError, expected_regex="Cannot modify locked TensorDict"
            ):
                td_unflatten = td_flatten.unflatten_keys(
                    inplace=inplace, separator=separator
                )
            return
        else:
            if locked:
                td.lock_()
            td_flatten = td.flatten_keys(inplace=inplace, separator=separator)
            td_unflatten = td_flatten.unflatten_keys(
                inplace=inplace, separator=separator
            )
        assert (td == td_unflatten).all()
        if inplace:
            assert td is td_unflatten

    @parametrize("td_type", TD_TYPES)
    def test_unlock(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        td.unlock_()
        assert not td.is_locked
        if td.device is not None:
            assert td.device.type == "cuda" or not td.is_shared()
        else:
            assert not td.is_shared()
        assert not td.is_memmap()

    @parametrize("squeeze_dim", [0, 1])
    @parametrize("td_type", TD_TYPES)
    def test_unsqueeze(self, td_type, squeeze_dim):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        td.unlock_()  # make sure that the td is not locked
        td_unsqueeze = torch.unsqueeze(td, dim=squeeze_dim)
        tensor = torch.ones_like(td.get("a").unsqueeze(squeeze_dim))
        if td_type in ("sub_td", "sub_td2"):
            td_unsqueeze.set_("a", tensor)
        else:
            td_unsqueeze.set("a", tensor)
        assert (td_unsqueeze.get("a") == tensor).all()

    @parametrize("clone", [True, False])
    @parametrize("td_type", TD_TYPES)
    def test_update(self, td_type, clone):
        td = getattr(self, td_type)
        td.unlock_()  # make sure that the td is not locked
        keys = set(td.keys())
        td.update({"x": torch.zeros(td.shape)}, clone=clone)
        assert set(td.keys()) == keys.union({"x"})
        # now with nested: using tuples for keys
        td.update({("somenested", "z"): torch.zeros(td.shape)})
        assert td["somenested"].shape == td.shape
        assert td["somenested", "z"].shape == td.shape
        td.update({("somenested", "zz"): torch.zeros(td.shape)})
        assert td["somenested"].shape == td.shape
        assert td["somenested", "zz"].shape == td.shape
        # now with nested: using nested dicts
        td["newnested"] = {"z": torch.zeros(td.shape)}
        keys = set(td.keys(True))
        assert ("newnested", "z") in keys
        td.update({"newnested": {"y": torch.zeros(td.shape)}}, clone=clone)
        keys = keys.union({("newnested", "y")})
        assert keys == set(td.keys(True))
        td.update(
            {
                ("newnested", "x"): torch.zeros(td.shape),
                ("newnested", "w"): torch.zeros(td.shape),
            },
            clone=clone,
        )
        keys = keys.union({("newnested", "x"), ("newnested", "w")})
        assert keys == set(td.keys(True))
        td.update({("newnested",): {"v": torch.zeros(td.shape)}}, clone=clone)
        keys = keys.union(
            {
                ("newnested", "v"),
            }
        )
        assert keys == set(td.keys(True))

        if td_type in ("sub_td", "sub_td2"):
            with self.assertRaisesRegex(
                ValueError, expected_regex="Tried to replace a tensordict with"
            ):
                td.update({"newnested": torch.zeros(td.shape)}, clone=clone)
        else:
            td.update({"newnested": torch.zeros(td.shape)}, clone=clone)
            assert isinstance(td["newnested"], torch.Tensor)

    @parametrize("td_type", TD_TYPES)
    def test_update_at_(self, td_type):
        td = getattr(self, td_type)
        td0 = td[1].clone().zero_()
        if td_type == "td_params":
            with self.assertRaisesRegex(
                RuntimeError, expected_regex="a view of a leaf Variable"
            ):
                td.update_at_(td0, 0)
            return
        td.update_at_(td0, 0)
        assert (td[0] == 0).all()

    @parametrize(
        "index", ["tensor1", "mask", "int", "range", "tensor2", "slice_tensor"]
    )
    @parametrize("td_type", TD_TYPES)
    def test_update_subtensordict(self, td_type, index):
        td = getattr(self, td_type)
        device = td.device
        if index == "mask":
            index = torch.zeros(td.shape[0], dtype=torch.bool, device=device)
            index[-1] = 1
        elif index == "int":
            index = td.shape[0] - 1
        elif index == "range":
            index = range(td.shape[0] - 1, td.shape[0])
        elif index == "tensor1":
            index = torch.tensor(td.shape[0] - 1, device=device)
        elif index == "tensor2":
            index = torch.tensor([td.shape[0] - 2, td.shape[0] - 1], device=device)
        elif index == "slice_tensor":
            index = (
                slice(None),
                torch.tensor([td.shape[1] - 2, td.shape[1] - 1], device=device),
            )

        sub_td = td._get_sub_tensordict(index)
        assert sub_td.shape == td.to_tensordict()[index].shape
        assert sub_td.shape == td[index].shape, (td, index)
        td0 = td[index]
        td0 = td0.to_tensordict()
        td0 = td0.apply(lambda x: x * 0 + 2)
        assert sub_td.shape == td0.shape
        if td_type == "td_params":
            with self.assertRaisesRegex(RuntimeError, expected_regex="a leaf Variable"):
                sub_td.update(td0)
            return
        sub_td.update(td0)
        assert (sub_td == 2).all()
        assert (td[index] == 2).all()

    @parametrize("td_type", TD_TYPES)
    def test_view(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        td.unlock_()  # make sure that the td is not locked
        td_view = td.view(-1)
        tensor = td.get("a")
        tensor = tensor.view(-1, tensor.numel() // math.prod(td.batch_size))
        tensor = torch.ones_like(tensor)
        td_view.set("a", tensor)
        assert (td_view.get("a") == tensor).all()
        assert (td_view.get("a") == 1).all()

    @parametrize("td_type", TD_TYPES)
    def test_where(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        device = td.device
        mask = torch.zeros(td.shape, dtype=torch.bool, device=device).bernoulli_()
        td_where = torch.where(mask, td, 0)
        for k in td.keys(True, True):
            assert (td_where.get(k)[~mask] == 0).all()
        td_where = torch.where(mask, td, torch.ones_like(td))
        for k in td.keys(True, True):
            assert (td_where.get(k)[~mask] == 1).all()
        td_where = td.clone()
        assert not td_where.is_locked, td_where
        if td_type in ("td_params",):
            with self.assertRaisesRegex(
                RuntimeError,
                expected_regex=re.escape(
                    "Cannot modify locked TensorDict. For in-place modification, consider using the `set_()` method and make sure the key is present."
                ),
            ):
                torch.where(mask, td, torch.ones_like(td), out=td_where)
            return
        else:
            torch.where(mask, td, torch.ones_like(td), out=td_where)
        for k in td.keys(True, True):
            assert (td_where.get(k)[~mask] == 1).all()

    @parametrize("td_type", TD_TYPES)
    def test_where_pad(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        # test with other empty td
        mask = torch.zeros(td.shape, dtype=torch.bool, device=td.device).bernoulli_()
        td_full = td
        td_empty = td_full.empty()
        result = td.where(mask, td_empty, pad=1)
        for v in result.values(True, True):
            assert (v[~mask] == 1).all()
        td_empty = td_full.empty()
        result = td_empty.where(~mask, td, pad=1)
        for v in result.values(True, True):
            assert (v[~mask] == 1).all()
        # with output
        td_out = td_full.empty()
        result = td.where(mask, td_empty, pad=1, out=td_out)
        for v in result.values(True, True):
            assert (v[~mask] == 1).all()
        if td_type not in ("td_params",):
            assert result is td_out
        td_out = td_full.empty()
        td_empty = td_full.empty()
        result = td_empty.where(~mask, td, pad=1, out=td_out)
        for v in result.values(True, True):
            assert (v[~mask] == 1).all()
        assert result is td_out

        with self.assertRaisesRegex(
            KeyError, expected_regex="not found and no pad value provided"
        ):
            td.where(mask, td_full.empty())
        with self.assertRaisesRegex(
            KeyError, expected_regex="not found and no pad value provided"
        ):
            td_full.empty().where(mask, td)

    @parametrize("td_type", TD_TYPES)
    def test_write_on_subtd(self, td_type):
        td = getattr(self, td_type)
        device = td.device
        sub_td = td._get_sub_tensordict(0)
        # should not work with td_params
        if td_type == "td_params":
            with self.assertRaisesRegex(
                RuntimeError, expected_regex="a view of a leaf"
            ):
                sub_td["a"] = torch.full((3, 2, 1, 5), 1.0, device=device)
            return
        sub_td["a"] = torch.full((3, 2, 1, 5), 1.0, device=device)
        assert (td["a"][0] == 1).all()

    @parametrize("td_type", TD_TYPES)
    def test_zero_(self, td_type):
        torch.manual_seed(1)
        td = getattr(self, td_type)
        if td_type == "td_params":
            with self.assertRaisesRegex(
                RuntimeError, expected_regex="a leaf Variable that requires grad"
            ):
                td.zero_()
            return
        new_td = td.zero_()
        assert new_td is td
        for k in td.keys():
            assert (td.get(k) == 0).all()


instantiate_parametrized_tests(TestTensorDicts)


class TestTensorDictMP(TestCase):
    @classmethod
    def _test_mp_remote_process(
        cls, worker_id, command_pipe_child, command_pipe_parent, tensordict
    ):
        command_pipe_parent.close()
        while True:
            cmd, val = command_pipe_child.recv()
            if cmd == "recv":
                b = tensordict.get("b")
                assert (b == val).all()
                command_pipe_child.send("done")
            elif cmd == "send":
                a = torch.ones(2) * val
                tensordict.set_("a", a)
                assert (
                    tensordict.get("a") == a
                ).all(), f'found {a} and {tensordict.get("a")}'
                command_pipe_child.send("done")
            elif cmd == "set_done":
                tensordict.set_("done", torch.ones(1, dtype=torch.bool))
                command_pipe_child.send("done")
            elif cmd == "set_undone_":
                tensordict.set_("done", torch.zeros(1, dtype=torch.bool))
                command_pipe_child.send("done")
            elif cmd == "update":
                tensordict.update_(
                    TensorDict(
                        source={"a": tensordict.get("a").clone() + 1},
                        batch_size=tensordict.batch_size,
                    )
                )
                command_pipe_child.send("done")
            elif cmd == "update_":
                tensordict.update_(
                    TensorDict(
                        source={"a": tensordict.get("a").clone() - 1},
                        batch_size=tensordict.batch_size,
                    )
                )
                command_pipe_child.send("done")

            elif cmd == "close":
                command_pipe_child.close()
                break

    @classmethod
    def _test_mp_driver_func(cls, tensordict, tensordict_unbind):
        procs = []
        children = []
        parents = []

        for i in range(2):
            command_pipe_parent, command_pipe_child = mp.Pipe()
            proc = mp.Process(
                target=cls._test_mp_remote_process,
                args=(i, command_pipe_child, command_pipe_parent, tensordict_unbind[i]),
            )
            proc.start()
            command_pipe_child.close()
            parents.append(command_pipe_parent)
            children.append(command_pipe_child)
            procs.append(proc)

        b = torch.ones(2, 1) * 10
        tensordict.set_("b", b)
        for i in range(2):
            parents[i].send(("recv", 10))
            is_done = parents[i].recv()
            assert is_done == "done"

        for i in range(2):
            parents[i].send(("send", i))
            is_done = parents[i].recv()
            assert is_done == "done"
        a = tensordict.get("a").clone()
        assert (a[0] == 0).all()
        assert (a[1] == 1).all()

        assert not tensordict.get("done").any()
        for i in range(2):
            parents[i].send(("set_done", i))
            is_done = parents[i].recv()
            assert is_done == "done"
        assert tensordict.get("done").all()

        for i in range(2):
            parents[i].send(("set_undone_", i))
            is_done = parents[i].recv()
            assert is_done == "done"
        assert not tensordict.get("done").any()

        a_prev = tensordict.get("a").clone().contiguous()
        for i in range(2):
            parents[i].send(("update_", i))
            is_done = parents[i].recv()
            assert is_done == "done"
        new_a = tensordict.get("a").clone().contiguous()
        torch.testing.assert_close(a_prev - 1, new_a)

        a_prev = tensordict.get("a").clone().contiguous()
        for i in range(2):
            parents[i].send(("update", i))
            is_done = parents[i].recv()
            assert is_done == "done"
        new_a = tensordict.get("a").clone().contiguous()
        torch.testing.assert_close(a_prev + 1, new_a)

        for i in range(2):
            parents[i].send(("close", None))
            procs[i].join()

    @parametrize(
        "td_type",
        [
            "memmap",
            "contiguous",
        ],
    )
    def test_mp(self, td_type):
        tensordict = TensorDict(
            source={
                "a": torch.randn(2, 2),
                "b": torch.randn(2, 1),
                "done": torch.zeros(2, 1, dtype=torch.bool),
            },
            batch_size=[2],
        )
        if td_type == "contiguous":
            tensordict = tensordict.share_memory_()
        elif td_type == "memmap":
            tensordict = tensordict.memmap_()
        else:
            raise NotImplementedError
        self._test_mp_driver_func(
            tensordict,
            (tensordict._get_sub_tensordict(0), tensordict._get_sub_tensordict(1))
            # tensordict,
            # tensordict.unbind(0),
        )


instantiate_parametrized_tests(TestTensorDictMP)


class TestUtils(TestCase):
    @parametrize(
        "ellipsis_index, expectation",
        [
            ((..., 0, ...), RuntimeError),
            ((0, ..., 0, ...), RuntimeError),
        ],
    )
    def test_convert_ellipsis_to_idx_invalid(self, ellipsis_index, expectation):
        torch.manual_seed(1)
        batch_size = [3, 4, 5, 6, 7]

        with self.assertRaises(expectation):
            convert_ellipsis_to_idx(ellipsis_index, batch_size)

    @parametrize(
        "ellipsis_index, expected_index",
        [
            (..., (slice(None), slice(None), slice(None), slice(None), slice(None))),
            ((0, ..., 0), (0, slice(None), slice(None), slice(None), 0)),
            ((..., 0), (slice(None), slice(None), slice(None), slice(None), 0)),
            ((0, ...), (0, slice(None), slice(None), slice(None), slice(None))),
            (
                (slice(1, 2), ...),
                (slice(1, 2), slice(None), slice(None), slice(None), slice(None)),
            ),
        ],
    )
    def test_convert_ellipsis_to_idx_valid(self, ellipsis_index, expected_index):
        torch.manual_seed(1)
        batch_size = [3, 4, 5, 6, 7]

        assert convert_ellipsis_to_idx(ellipsis_index, batch_size) == expected_index

    @parametrize(
        "idx",
        [
            (slice(None),),
            slice(None),
            (3, 4),
            (3, slice(None), slice(2, 2, 2)),
            (torch.tensor([1, 2, 3]),),
            ([1, 2, 3]),
            (
                torch.tensor([1, 2, 3]),
                torch.tensor([2, 3, 4]),
                torch.tensor([0, 10, 2]),
                torch.tensor([2, 4, 1]),
            ),
            torch.zeros(10, 7, 11, 5, dtype=torch.bool).bernoulli_(),
            torch.zeros(10, 7, 11, dtype=torch.bool).bernoulli_(),
            (0, torch.zeros(7, dtype=torch.bool).bernoulli_()),
        ],
    )
    def test_getitem_batch_size(self, idx):
        shape = [10, 7, 11, 5]
        shape = torch.Size(shape)
        mocking_tensor = torch.zeros(*shape)
        expected_shape = mocking_tensor[idx].shape
        resulting_shape = _getitem_batch_size(shape, idx)
        assert expected_shape == resulting_shape, (idx, expected_shape, resulting_shape)

    @parametrize("device", get_available_devices())
    def test_squeeze(self, device):
        torch.manual_seed(1)
        d = {
            "key1": torch.randn(4, 5, 6, device=device),
            "key2": torch.randn(4, 5, 10, device=device),
        }
        td1 = TensorDict(batch_size=(4, 5), source=d)
        td2 = torch.unsqueeze(td1, dim=1)
        assert td2.batch_size == torch.Size([4, 1, 5])

        td1b = torch.squeeze(td2, dim=1)
        assert td1b.batch_size == td1.batch_size


instantiate_parametrized_tests(TestUtils)


class TestTensorDictVmap(TestCase):
    def test_vmap(self):
        td = TensorDict({"a": torch.zeros(3, 4, 5)}, batch_size=[3, 4])
        ones = torch.ones(4, 5)
        td_out = torch.vmap(
            lambda td, one: td.set("a", td.get("a") + one),
            in_dims=(0, None),
            out_dims=(0,),
        )(td, ones)
        assert td_out.shape == torch.Size([3, 4]), td_out.shape

        td_out = torch.vmap(
            lambda td, one: td.set("a", td.get("a") + one),
            in_dims=(0, None),
            out_dims=(1,),
        )(td, ones)
        assert td_out.shape == torch.Size([4, 3]), td_out.shape

        ones = torch.ones(3, 5)
        td_out = torch.vmap(
            lambda td, one: td.set("a", td.get("a") + one),
            in_dims=(1, None),
            out_dims=(1,),
        )(td, ones)
        assert td_out.shape == torch.Size([3, 4]), td_out.shape

        td_out = torch.vmap(
            lambda td, one: td.set("a", td.get("a") + one),
            in_dims=(1, None),
            out_dims=(0,),
        )(td, ones)
        assert td_out.shape == torch.Size([4, 3]), td_out.shape


class TestPyTree(TestCase):
    def test_pytree_map(self):
        td = TensorDict({"a": {"b": {"c": 1}, "d": 1}, "e": 1}, [])
        td = tree_map(lambda x: x + 1, td)
        assert (td == 2).all()

    def test_pytree_map_batch(self):
        td = TensorDict(
            {
                "a": TensorDict(
                    {
                        "b": TensorDict({"c": torch.ones(2, 3, 4)}, [2, 3]),
                        "d": torch.ones(2),
                    },
                    [2],
                ),
                "e": 1,
            },
            [],
        )
        td = tree_map(lambda x: x + 1, td)
        assert (td == 2).all()
        assert td.shape == torch.Size([])
        assert td["a"].shape == torch.Size([2])
        assert td["a", "b"].shape == torch.Size([2, 3])
        assert td["a", "b", "c"].shape == torch.Size([2, 3, 4])

    def test_pytree_vs_apply(self):
        td = TensorDict(
            {
                "a": TensorDict(
                    {
                        "b": TensorDict({"c": torch.ones(2, 3, 4)}, [2, 3]),
                        "d": torch.ones(2),
                    },
                    [2],
                ),
                "e": 1,
            },
            [],
        )
        td_pytree = tree_map(lambda x: x + 1, td)
        td_apply = td.apply(lambda x: x + 1)
        assert (td_apply == td_pytree).all()
        for v1, v2 in zip(td_pytree.values(True), td_apply.values(True)):
            # recursively checks the shape, including for the nested tensordicts
            assert v1.shape == v2.shape


if __name__ == "__main__":
    run_tests()

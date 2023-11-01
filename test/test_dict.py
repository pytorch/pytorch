from __future__ import annotations

import re

import torch.cuda
from torch import multiprocessing as mp
from torch import nn
from torch.dict import TensorDict, TensorDictBase, TensorDictParams, pad, \
    pad_sequence
from torch.dict.base import is_tensor_collection
from torch.dict.utils import convert_ellipsis_to_idx
from torch.dict.tensordict import _getitem_batch_size
from torch.testing._internal.common_utils import TestCase, run_tests, \
    parametrize, instantiate_parametrized_tests


class raises:
    def __init__(
        self,
        _self: TestCase,
        error_type: Exception,
        match: str | None = None
    ):
        self._self = _self
        self.error_type = error_type
        self.match = match

    def __enter__(self):
        self._ctx = self._self.assertRaises(self.error_type)
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        result = self._ctx.__exit__(exc_type, exc_val, exc_tb)
        if self.match is not None:
            err_str = self._ctx.exception.args[0]
            self._self.assertTrue(re.match(self.match, err_str))
        return result


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
                    {
                        "e": torch.randn(4, 3, 2, 1, 2)
                    }, batch_size=[4, 3, 2, 1, 2]
                )
            },
            batch_size=[4, 3, 2, 1],
        )

    @property
    def td_params(self):
        return TensorDictParams(self.td_nested)

    @parametrize(
        "td_type",
        ["td_device", "td_no_device", "td_nested", "td_params"]
    )
    def test_creation(self, td_type):
        self.assertIsNot(getattr(self, td_type), None)

    @parametrize(
        "td_type",
        ["td_device", "td_no_device", "td_nested", "td_params"]
    )
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
            {"a": torch.randn(3, 4, 5, 6), "b": torch.randn(3, 4, 5)},
            batch_size=[3, 4]
        )
        # smoke-test
        td.batch_size = torch.Size([3])

        # test with list
        td.batch_size = [3]

        # test with tuple
        td.batch_size = (3,)

        # incompatible size
        with raises(
            self,
            RuntimeError,
            match=re.escape(
                "the tensor a has shape torch.Size([3, 4, 5, 6]) which is incompatible with the batch-size torch.Size([3, 5])"
            )
        ):
            td.batch_size = [3, 5]

        # test set
        td.set("c", torch.randn(3))

        # test index
        td[torch.tensor([1, 2])]
        td[:]
        td[[1, 2]]
        with raises(
            self,
            IndexError,
            match="too many indices for tensor of dimension 1",
        ):
            td[:, 0]

        # test a greater batch_size
        td = TensorDict(
            {"a": torch.randn(3, 4, 5, 6), "b": torch.randn(3, 4, 5)},
            batch_size=[3, 4]
        )
        td.batch_size = torch.Size([3, 4, 5])

        td.set("c", torch.randn(3, 4, 5, 6))
        with raises(
            self,
            RuntimeError,
            match=re.escape(
                "batch dimension mismatch, got self.batch_size=torch.Size([3, 4, 5]) and value.shape=torch.Size([3, 4, 2])"
            )
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
            {"a": TensorDict({"b": torch.rand(1, 2)}, [1, 2]),
             "c": torch.rand(1)}, [1]
        )
        with raises(
            self,
            NotImplementedError,
            match="TensorDict does not support membership checks with the `in` keyword",
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
        td.set_(
            "a",
            torch.randn(4, 5)
        )  # No exception because set_ ignores the lock

        with raises(
            self,
            RuntimeError,
            match="Cannot modify locked TensorDict"
        ):
            td.set("a", torch.randn(4, 5))

        with raises(
            self,
            RuntimeError,
            match="Cannot modify locked TensorDict"
        ):
            td.set("b", torch.randn(4, 5))

        with raises(
            self,
            RuntimeError,
            match="Cannot modify locked TensorDict"
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
            other_td = TensorDict(
                {"a": torch.randn(10), "b": torch.ones(1)},
                []
            )
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
            {f"a{separator}b": torch.zeros(3),
             "a": {"b": {"c": torch.zeros(3)}}}, []
        )

        with raises(
            self,
            KeyError,
            match=re.escape("Flattening keys in tensordict causes keys [('a', 'b', 'c')] to collide.")
        ):
            td1.flatten_keys(separator)

        with raises(
            self,
            KeyError,
            match=re.escape("Flattening keys in tensordict causes keys [('a', 'b')] to collide.")
        ):
            td2.flatten_keys(separator)

        with raises(
            self,
            KeyError,
            match=re.escape("Flattening keys in tensordict causes keys [('a', 'b', 'c')] to collide.")
        ):
            td3.flatten_keys(separator)

        with raises(
            self,
            KeyError,
            match=re.escape(
                "Unflattening key(s) in tensordict will override existing unflattened key"
            ),
        ):
            td1.unflatten_keys(separator)

        with raises(
            self,
            KeyError,
            match=re.escape(
                "Unflattening key(s) in tensordict will override existing unflattened key"
            ),
        ):
            td2.unflatten_keys(separator)

        with raises(
            self,
            KeyError,
            match=re.escape(
                "Unflattening key(s) in tensordict will override existing unflattened key"
            ),
        ):
            td3.unflatten_keys(separator)

        with raises(
            self,
            KeyError,
            match=re.escape(
                "Unflattening key(s) in tensordict will override existing unflattened key"
            ),
        ):
            td4.unflatten_keys(separator)

        with raises(
            self,
            KeyError,
            match=re.escape(
                "Unflattening key(s) in tensordict will override existing unflattened key"
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
            with raises(
                self,
                ValueError,
                match="Cannot pass both batch_size and batch_dims"
            ):
                TensorDict.from_dict(
                    data,
                    batch_size=batch_size,
                    batch_dims=batch_dims,
                    device=device
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

    @parametrize(
        "td_type",
        ["td_device", "td_no_device", "td_nested", "td_params"]
    )
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

        with raises(
            self,
            TypeError,
            match=re.escape(
                "Nested membership checks with tuples of strings is only supported when setting"
            )
        ):
            ("a", "b", "c") in tensordict.keys()  # noqa: B015

        with raises(
            self,
            TypeError,
            match="TensorDict keys are always strings."
        ):
            42 in tensordict.keys()  # noqa: B015

        with raises(
            self,
            TypeError,
            match="TensorDict keys are always strings."
        ):
            ("a", 42) in tensordict.keys()  # noqa: B015

        keys = set(tensordict.keys())
        keys_nested = set(tensordict.keys(include_nested=True))

        assert keys == {"a"}
        assert keys_nested == {"a", ("a", "b"), ("a", "b", "c")}

        leaves = set(tensordict.keys(leaves_only=True))
        leaves_nested = set(
            tensordict.keys(include_nested=True, leaves_only=True)
        )

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
            device="cpu"
        )
        td_memmap = td.clone().memmap_()
        assert (td == td_memmap).all()

        assert (td == td_memmap.apply(lambda x: x.clone())).all()
        if device.type == "cuda":
            td = td.pin_memory()
            td_memmap = td.clone().memmap_()
            td_memmap_pm = td_memmap.apply(lambda x: x.clone()).pin_memory()
            assert (td.pin_memory().to(device) == td_memmap_pm.to(
                device
            )).all()

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
        td.set_(
            "a",
            torch.randn(4, 5)
        )  # No exception because set_ ignores the lock

        with raises(
            self,
            RuntimeError,
            match="Cannot modify locked TensorDict"
        ):
            td.set("a", torch.randn(4, 5))

        with raises(
            self,
            RuntimeError,
            match="Cannot modify locked TensorDict"
        ):
            td.set("b", torch.randn(4, 5))

        with raises(
            self,
            RuntimeError,
            match="Cannot modify locked TensorDict"
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

        padded_td = pad(
            td,
            [dim0_left, dim0_right, dim1_left, dim1_right],
            value=0.0
        )

        expected_a = torch.cat(
            [torch.ones(3, 4, 1), torch.zeros(1, 4, 1)],
            dim=0
        )
        expected_a = torch.cat([expected_a, torch.zeros(4, 2, 1)], dim=1)

        assert padded_td["a"].shape == (4, 6, 1)
        assert padded_td["b"].shape == (4, 6, 1, 1)
        assert torch.equal(padded_td["a"], expected_a)
        padded_td._check_batch_size()

    @parametrize("batch_first", [True, False])
    @parametrize("make_mask", [True, False])
    def test_pad_sequence(self, batch_first, make_mask):
        list_td = [
            TensorDict(
                {"a": torch.ones((2,)), ("b", "c"): torch.ones((2, 3))},
                [2]
            ),
            TensorDict(
                {"a": torch.ones((4,)), ("b", "c"): torch.ones((4, 3))},
                [4]
            ),
        ]
        padded_td = pad_sequence(
            list_td,
            batch_first=batch_first,
            return_mask=make_mask
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

        with raises(self, ValueError):
            td2 = td1.permute(1, 1, 0)
            _ = td2.shape

        with raises(self, ValueError):
            td2 = td1.permute(3, 2, 1, 0)
            _ = td2.shape

        with raises(self, ValueError):
            td2 = td1.permute(2, -1, 0)
            _ = td2.shape

        with raises(self, ValueError):
            td2 = td1.permute(2, 3, 0)
            _ = td2.shape

        with raises(self, ValueError):
            td2 = td1.permute(2, -4, 0)
            _ = td2.shape

        with raises(self, ValueError):
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
        td1 = TensorDict(batch_size=(20, 6), source=d).view(4, 5, 6).permute(
            2,
            1,
            0
        )
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
                        requires_grad=True if i == 5 else False
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
        with raises(
            self,
            RuntimeError,
            match="TensorDict with empty batch size is not splittable"
        ):
            td.split(1, 0)

        td = TensorDict({}, [3, 2])

        # Test invalid split_size input
        with raises(
            self,
            TypeError,
            match=re.escape(
                "split(): argument 'split_size' must be int or list of ints"
                )
            ):
            td.split("1", 0)
        with raises(
            self,
            TypeError,
            match=re.escape(
                "split(): argument 'split_size' must be int or list of ints"
                )
            ):
            td.split(["1", 2], 0)

        # Test invalid split_size sum
        with raises(
            self,
            RuntimeError,
            match=re.escape(
                "Split method expects split_size to sum exactly to 3 (tensor's size at dimension 0), but got split_size=[]"
                )
        ):
            td.split([], 0)

        with raises(
            self,
            RuntimeError,
            match=re.escape(
                "Split method expects split_size to sum exactly to 3 (tensor's size at dimension 0), but got split_size=[1, 1]"
                )
        ):
            td.split([1, 1], 0)

        # Test invalid dimension input
        with raises(
            self,
            IndexError,
            match=re.escape("Dimension out of range")
            ):
            td.split(1, 2)
        with raises(
            self,
            IndexError,
            match=re.escape("Dimension out of range")
            ):
            td.split(1, -3)

    def test_split_with_negative_dim(self):
        td = TensorDict(
            {"a": torch.zeros(5, 4, 2, 1), "b": torch.zeros(5, 4, 1)},
            [5, 4]
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
        with raises(
            self,
            RuntimeError,
            match=re.escape(
                "Calling `_SubTensorDict.set(key, value, inplace=False)` is prohibited for existing tensors. Consider calling _SubTensorDict.set_(...) or cloning your tensordict first."
                )
        ):
            std_control.set("key1", torch.randn(1, device=device))
        with raises(
            self,
            RuntimeError,
            match=re.escape(
                "Calling `_SubTensorDict.set(key, value, inplace=False)` is prohibited for existing tensors. Consider calling _SubTensorDict.set_(...) or cloning your tensordict first."
                )
        ):
            std_control.set(
                "key2",
                torch.randn(6, device=device, dtype=torch.double)
            )

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

        with raises(
            self,
            RuntimeError,
            match='tensors on different devices at key "sub" / "a"'
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
        with raises(self, RuntimeError):
            td.set("key1", torch.randn(5, 5, device=device))

        # robust to dtype casting
        td.set_("key1", torch.ones(4, 5, device=device, dtype=torch.double))
        assert (td.get("key1") == 1).all()

        # robust to device casting
        td.set(
            "key_device",
            torch.ones(4, 5, device="cpu", dtype=torch.double)
        )
        assert td.get("key_device").device == torch.device(device)

        with raises(
            self,
            KeyError,
            match=re.escape(
                "key \"smartypants\" not found in TensorDict with keys"
            )
        ):
            td.set_(
                "smartypants",
                torch.ones(4, 5, device="cpu", dtype=torch.double)
            )
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
            "key1",
            torch.zeros(4, 5, dtype=torch.double, device=device),
            inplace=True
        )
        assert (td.get("key1") == 0).all()
        td.set(
            "key1",
            torch.randn(4, 5, 1, 2, dtype=torch.double, device=device),
            inplace=False,
        )
        assert td["key1"].shape == td._tensordict["key1"].shape

    def test_unbind_batchsize(self):
        td = TensorDict(
            {"a": TensorDict({"b": torch.zeros(2, 3)}, [2, 3])},
            [2]
        )
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


instantiate_parametrized_tests(TestTensorDicts)


class TestTensorDictMP(TestCase):
    @classmethod
    def _test_mp_remote_process(
        cls,
        worker_id,
        command_pipe_child,
        command_pipe_parent,
        tensordict
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
                args=(i, command_pipe_child, command_pipe_parent,
                      tensordict_unbind[i]),
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
            (tensordict._get_sub_tensordict(0),
             tensordict._get_sub_tensordict(1))
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
    def test_convert_ellipsis_to_idx_invalid(
        self,
        ellipsis_index,
        expectation
    ):
        torch.manual_seed(1)
        batch_size = [3, 4, 5, 6, 7]

        with self.assertRaises(expectation):
            convert_ellipsis_to_idx(ellipsis_index, batch_size)

    @parametrize(
        "ellipsis_index, expected_index",
        [
            (..., (
                slice(None), slice(None), slice(None), slice(None),
                slice(None))),
            ((0, ..., 0), (0, slice(None), slice(None), slice(None), 0)),
            (
                (..., 0),
                (slice(None), slice(None), slice(None), slice(None), 0)),
            (
                (0, ...),
                (0, slice(None), slice(None), slice(None), slice(None))),
            (
                (slice(1, 2), ...),
                (slice(1, 2), slice(None), slice(None), slice(None),
                 slice(None)),
            ),
        ],
    )
    def test_convert_ellipsis_to_idx_valid(
        self,
        ellipsis_index,
        expected_index
    ):
        torch.manual_seed(1)
        batch_size = [3, 4, 5, 6, 7]

        assert convert_ellipsis_to_idx(
            ellipsis_index,
            batch_size
        ) == expected_index

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
        assert expected_shape == resulting_shape, (
            idx, expected_shape, resulting_shape)

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
        td_out = torch.vmap(lambda td, one: td.set("a", td.get("a") + one), in_dims=(0, None), out_dims=(0,))(td, ones)
        assert td_out.shape == torch.Size([3, 4]), td_out.shape

        td_out = torch.vmap(lambda td, one: td.set("a", td.get("a") + one), in_dims=(0, None), out_dims=(1,))(td, ones)
        assert td_out.shape == torch.Size([4, 3]), td_out.shape

        ones = torch.ones(3, 5)
        td_out = torch.vmap(lambda td, one: td.set("a", td.get("a") + one), in_dims=(1, None), out_dims=(1,))(td, ones)
        assert td_out.shape == torch.Size([3, 4]), td_out.shape

        td_out = torch.vmap(lambda td, one: td.set("a", td.get("a") + one), in_dims=(1, None), out_dims=(0,))(td, ones)
        assert td_out.shape == torch.Size([4, 3]), td_out.shape


if __name__ == '__main__':
    run_tests()

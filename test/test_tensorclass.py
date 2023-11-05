from __future__ import annotations

import dataclasses
import inspect
import os
import pickle
import re
from multiprocessing import Pool
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional, Tuple, Union

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TemporaryDirectoryName,
)

try:
    import torchsnapshot

    _has_torchsnapshot = True
    TORCHSNAPSHOT_ERR = ""
except ImportError as err:
    _has_torchsnapshot = False
    TORCHSNAPSHOT_ERR = str(err)

from tensordict import (
    is_tensorclass,
    LazyStackedTensorDict,
    MemoryMappedTensor,
    tensorclass,
    TensorDict,
)
from tensordict.tensordict import (
    _PermutedTensorDict,
    _ViewedTensorDict,
    assert_allclose_td,
    TensorDictBase,
)
from torch import Tensor
from torch.testing._internal.common_utils import TestCase


def get_available_devices():
    devices = [torch.device("cpu")]
    n_cuda = torch.cuda.device_count()
    if n_cuda > 0:
        for i in range(n_cuda):
            devices += [torch.device(f"cuda:{i}")]
    return devices


class TestTensorClass(TestCase):
    class MyData:
        X: torch.Tensor
        y: torch.Tensor
        z: str

        def stuff(self):
            return self.X + self.y

    # this slightly convoluted construction of MyData allows us to check that instances of
    # the tensorclass are instances of the original class.
    MyDataUndecorated, MyData = MyData, tensorclass(MyData)

    @tensorclass
    class MyData2:
        X: torch.Tensor
        y: torch.Tensor
        z: list

    def test_dataclass(self):
        data = self.MyData(
            X=torch.ones(3, 4, 5),
            y=torch.zeros(3, 4, 5, dtype=torch.bool),
            z="test_tensorclass",
            batch_size=[3, 4],
        )
        assert dataclasses.is_dataclass(data)

    def test_type(self):
        data = self.MyData(
            X=torch.ones(3, 4, 5),
            y=torch.zeros(3, 4, 5, dtype=torch.bool),
            z="test_tensorclass",
            batch_size=[3, 4],
        )
        assert isinstance(data, self.MyData)
        assert is_tensorclass(data)
        assert is_tensorclass(self.MyData)
        # we get an instance of the user defined class, not a dynamically defined subclass
        assert type(data) is self.MyDataUndecorated

    def test_signature(self):
        sig = inspect.signature(self.MyData)
        assert list(sig.parameters) == ["X", "y", "z", "batch_size", "device"]

        with self.assertRaisesRegex(
            TypeError,
            expected_regex=re.escape(
                "MyData.__init__() missing 3 required positional arguments: 'X', 'y', 'z'"
            ),
        ):
            self.MyData(batch_size=[10])

        with self.assertRaisesRegex(
            TypeError,
            expected_regex=re.escape(
                "MyData.__init__() missing 2 required positional arguments: 'y', 'z'"
            ),
        ):
            self.MyData(X=torch.rand(10), batch_size=[10])

        with self.assertRaisesRegex(
            TypeError,
            expected_regex=re.escape(
                "MyData.__init__() missing 1 required positional argument: 'z'"
            ),
        ):
            self.MyData(
                X=torch.rand(10), y=torch.rand(10), batch_size=[10], device="cpu"
            )

        # if all positional arguments are specified, ommitting batch_size gives error
        with self.assertRaisesRegex(
            TypeError,
            expected_regex=re.escape(
                "MyData.__init__() missing 1 required keyword-only argument: 'batch_size'"
            ),
        ):
            self.MyData(X=torch.rand(10), y=torch.rand(10))

        # all positional arguments + batch_size is fine
        self.MyData(
            X=torch.rand(10), y=torch.rand(10), z="test_tensorclass", batch_size=[10]
        )

    @parametrize("device", get_available_devices())
    def test_device(self, device):
        data = self.MyData(
            X=torch.ones(3, 4, 5),
            y=torch.zeros(3, 4, 5, dtype=torch.bool),
            z="test_tensorclass",
            batch_size=[3, 4],
            device=device,
        )
        assert data.device == device
        assert data.X.device == device
        assert data.y.device == device

        with self.assertRaisesRegex(
            AttributeError, expected_regex="'str' object has no attribute 'device'"
        ):
            assert data.z.device == device

        with self.assertRaisesRegex(
            RuntimeError,
            expected_regex="device cannot be set using tensorclass.device = device",
        ):
            data.device = torch.device("cpu")

    def test_banned_types(self):
        @tensorclass
        class MyAnyClass:
            subclass: Any = None

        data = MyAnyClass(subclass=torch.ones(3, 4), batch_size=[3])
        assert data.subclass is not None

        @tensorclass
        class MyOptAnyClass:
            subclass: Optional[Any] = None

        data = MyOptAnyClass(subclass=torch.ones(3, 4), batch_size=[3])
        assert data.subclass is not None

        @tensorclass
        class MyUnionAnyClass:
            subclass: Union[Any] = None

        data = MyUnionAnyClass(subclass=torch.ones(3, 4), batch_size=[3])
        assert data.subclass is not None

        @tensorclass
        class MyUnionAnyTDClass:
            subclass: Union[Any, TensorDict] = None

        data = MyUnionAnyTDClass(subclass=torch.ones(3, 4), batch_size=[3])
        assert data.subclass is not None

        @tensorclass
        class MyOptionalClass:
            subclass: Optional[TensorDict] = None

        data = MyOptionalClass(subclass=TensorDict({}, [3]), batch_size=[3])
        assert data.subclass is not None

        data = MyOptionalClass(subclass=torch.ones(3), batch_size=[3])
        assert data.subclass is not None

        @tensorclass
        class MyUnionClass:
            subclass: Union[MyOptionalClass, TensorDict] = None

        data = MyUnionClass(
            subclass=MyUnionClass._from_tensordict(TensorDict({}, [3])), batch_size=[3]
        )
        assert data.subclass is not None

    def test_attributes(self):
        X = torch.ones(3, 4, 5)
        y = torch.zeros(3, 4, 5, dtype=torch.bool)
        batch_size = [3, 4]
        z = "test_tensorclass"
        tensordict = TensorDict(
            {
                "X": X,
                "y": y,
            },
            batch_size=[3, 4],
        )

        data = self.MyData(X=X, y=y, z=z, batch_size=batch_size)

        equality_tensordict = data._tensordict == tensordict

        assert torch.equal(data.X, X)
        assert torch.equal(data.y, y)
        assert data.batch_size == torch.Size(batch_size)
        assert equality_tensordict.all()
        assert equality_tensordict.batch_size == torch.Size(batch_size)
        assert data.z == z

    def test_disallowed_attributes(self):
        with self.assertRaisesRegex(
            AttributeError,
            expected_regex="Attribute name reshape can't be used with @tensorclass",
        ):

            @tensorclass
            class MyInvalidClass:
                x: torch.Tensor
                y: torch.Tensor
                reshape: torch.Tensor

    def test_batch_size(self):
        myc = self.MyData(
            X=torch.rand(2, 3, 4),
            y=torch.rand(2, 3, 4, 5),
            z="test_tensorclass",
            batch_size=[2, 3],
        )

        assert myc.batch_size == torch.Size([2, 3])
        assert myc.X.shape == torch.Size([2, 3, 4])

        myc.batch_size = torch.Size([2])

        assert myc.batch_size == torch.Size([2])
        assert myc.X.shape == torch.Size([2, 3, 4])

    def test_len(self):
        myc = self.MyData(
            X=torch.rand(2, 3, 4),
            y=torch.rand(2, 3, 4, 5),
            z="test_tensorclass",
            batch_size=[2, 3],
        )
        assert len(myc) == 2

        myc2 = self.MyData(
            X=torch.rand(2, 3, 4),
            y=torch.rand(2, 3, 4, 5),
            z="test_tensorclass",
            batch_size=[],
        )
        assert len(myc2) == 0

    def test_indexing(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: list
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        z = ["a", "b", "c"]
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)

        assert data[:2].batch_size == torch.Size([2, 4])
        assert data[:2].X.shape == torch.Size([2, 4, 5])
        assert (data[:2].X == X[:2]).all()
        assert isinstance(data[:2].y, type(data_nest))

        # Nested tensors all get indexed
        assert (data[:2].y.X == X[:2]).all()
        assert data[:2].y.batch_size == torch.Size([2, 4])
        assert data[1].batch_size == torch.Size([4])
        assert data[1][1].batch_size == torch.Size([])

        # Non-tensor data won't get indexed
        assert data[1].z == data[2].z == data[:2].z == z

        with self.assertRaisesRegex(
            RuntimeError,
            expected_regex="indexing a tensordict with td.batch_dims==0 is not permitted",
        ):
            data[1][1][1]

        with self.assertRaisesRegex(
            ValueError, expected_regex="Invalid indexing arguments."
        ):
            data["X"]

    def test_setitem(self):
        data = self.MyData(
            X=torch.ones(3, 4, 5),
            y=torch.zeros(3, 4, 5),
            z="test_tensorclass",
            batch_size=[3, 4],
        )

        x = torch.randn(3, 4, 5)
        y = torch.ones(3, 4, 5)
        z = "test_tensorclass"
        batch_size = [3, 4]
        data2 = self.MyData(X=x, y=y, z=z, batch_size=batch_size)
        data3 = self.MyData(X=y, y=x, z=z, batch_size=batch_size)

        # Testing the data before setting
        assert (data[:2].X == torch.ones(2, 4, 5)).all()
        assert (data[:2].y == torch.zeros(2, 4, 5)).all()
        assert data[:2].z == "test_tensorclass"
        assert (data[[1, 2]].X == torch.ones(5)).all()

        # Setting the item and testing post setting the item
        data[:2] = data2[:2].clone()
        assert (data[:2].X == data2[:2].X).all()
        assert (data[:2].y == data2[:2].y).all()
        assert data[:2].z == z

        data[[1, 2]] = data3[[1, 2]].clone()
        assert (data[[1, 2]].X == data3[[1, 2]].X).all()
        assert (data[[1, 2]].y == data3[[1, 2]].y).all()
        assert data[[1, 2]].z == z

        data[:, [1, 2]] = data2[:, [1, 2]].clone()
        assert (data[:, [1, 2]].X == data2[:, [1, 2]].X).all()
        assert (data[:, [1, 2]].y == data[:, [1, 2]].y).all()
        assert data[:, [1, 2]].z == z

        with self.assertRaisesRegex(
            RuntimeError, expected_regex="indexed destination TensorDict batch size is"
        ):
            data[:, [1, 2]] = data.clone()

        # Negative testcase for non-tensor data
        z = "test_bluff"
        data2 = self.MyData(X=x, y=y, z=z, batch_size=batch_size)
        with self.assertWarnsRegex(
            UserWarning,
            expected_regex="Meta data at 'z' may or may not be equal, this may result in undefined behaviours",
        ):
            data[1] = data2[1]

        # Validating nested test cases
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: list
            y: "MyDataNested" = None

        X = torch.randn(3, 4, 5)
        z = ["a", "b", "c"]
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
        X2 = torch.ones(3, 4, 5)
        data_nest2 = MyDataNested(X=X2, z=z, batch_size=batch_size)
        data2 = MyDataNested(X=X2, y=data_nest2, z=z, batch_size=batch_size)
        data[:2] = data2[:2].clone()
        assert (data[:2].X == data2[:2].X).all()
        assert (data[:2].y.X == data2[:2].y.X).all()
        assert data[:2].z == z

        # Negative Scenario
        data3 = MyDataNested(X=X2, y=data_nest2, z=["e", "f"], batch_size=batch_size)
        with self.assertWarnsRegex(
            UserWarning,
            expected_regex="Meta data at 'z' may or may not be equal, this may result in undefined behaviours",
        ):
            data[:2] = data3[:2]

    def test_setitem_memmap(self):
        # regression test PR #203
        # We should be able to set tensors items with MemmapTensors and viceversa
        @tensorclass
        class MyDataMemMap1:
            x: torch.Tensor
            y: MemoryMappedTensor

        data1 = MyDataMemMap1(
            x=torch.zeros(3, 4, 5),
            y=MemoryMappedTensor.from_tensor(torch.zeros(3, 4, 5)),
            batch_size=[3, 4],
        )

        data2 = MyDataMemMap1(
            x=MemoryMappedTensor.from_tensor(torch.ones(3, 4, 5)),
            y=torch.ones(3, 4, 5),
            batch_size=[3, 4],
        )

        data1[:2] = data2[:2]
        assert (data1[:2] == 1).all()
        assert (data1.x[:2] == 1).all()
        assert (data1.y[:2] == 1).all()
        data2[2:] = data1[2:]
        assert (data2[2:] == 0).all()
        assert (data2.x[2:] == 0).all()
        assert (data2.y[2:] == 0).all()

    def test_setitem_other_cls(self):
        @tensorclass
        class MyData1:
            x: torch.Tensor
            y: MemoryMappedTensor

        data1 = MyData1(
            x=torch.zeros(3, 4, 5),
            y=MemoryMappedTensor.from_tensor(torch.zeros(3, 4, 5)),
            batch_size=[3, 4],
        )

        # Set Item should work for other tensorclass
        @tensorclass
        class MyData2:
            x: MemoryMappedTensor
            y: torch.Tensor

        data_other_cls = MyData2(
            x=MemoryMappedTensor.from_tensor(torch.ones(3, 4, 5)),
            y=torch.ones(3, 4, 5),
            batch_size=[3, 4],
        )
        data1[:2] = data_other_cls[:2]
        data_other_cls[2:] = data1[2:]

        # Set Item should raise if other tensorclass with different members
        @tensorclass
        class MyData3:
            x: MemoryMappedTensor
            z: torch.Tensor

        data_wrong_cls = MyData3(
            x=MemoryMappedTensor.from_tensor(torch.ones(3, 4, 5)),
            z=torch.ones(3, 4, 5),
            batch_size=[3, 4],
        )
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="__setitem__ is only allowed for same-class or compatible class .* assignment",
        ):
            data1[:2] = data_wrong_cls[:2]
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="__setitem__ is only allowed for same-class or compatible class .* assignment",
        ):
            data_wrong_cls[2:] = data1[2:]

    @parametrize(
        "broadcast_type",
        ["scalar", "tensor", "tensordict", "maptensor"],
    )
    def test_setitem_broadcast(self, broadcast_type):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: list
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        z = ["a", "b", "c"]
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)

        if broadcast_type == "scalar":
            val = 0
        elif broadcast_type == "tensor":
            val = torch.zeros(4, 5)
        elif broadcast_type == "tensordict":
            val = TensorDict({"X": torch.zeros(2, 4, 5)}, batch_size=[2, 4])
        elif broadcast_type == "maptensor":
            val = MemoryMappedTensor.from_tensor(torch.zeros(4, 5))

        data[:2] = val
        assert (data[:2] == 0).all()
        assert (data.X[:2] == 0).all()
        assert (data.y.X[:2] == 0).all()

    def test_stack(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        z = "test_tensorclass"
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data1 = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
        data2 = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)

        stacked_tc = torch.stack([data1, data2], 0)
        assert type(stacked_tc) is type(data1)
        assert isinstance(stacked_tc.y, type(data1.y))
        assert stacked_tc.X.shape == torch.Size([2, 3, 4, 5])
        assert stacked_tc.y.X.shape == torch.Size([2, 3, 4, 5])
        assert (stacked_tc.X == 1).all()
        assert (stacked_tc.y.X == 1).all()
        assert isinstance(stacked_tc._tensordict, LazyStackedTensorDict)
        assert isinstance(stacked_tc.y._tensordict, LazyStackedTensorDict)
        assert stacked_tc.z == stacked_tc.y.z == z

        # Testing negative scenarios
        y = torch.zeros(3, 4, 5, dtype=torch.bool)
        data3 = self.MyData(X=X, y=y, z=z, batch_size=batch_size)

        with self.assertRaisesRegex(
            TypeError,
            expected_regex=("Multiple dispatch failed|no implementation found"),
        ):
            torch.stack([data1, data3], dim=0)

    def test_cat(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        z = "test_tensorclass"
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data1 = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
        data2 = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)

        catted_tc = torch.cat([data1, data2], 0)
        assert type(catted_tc) is type(data1)
        assert isinstance(catted_tc.y, type(data1.y))
        assert catted_tc.X.shape == torch.Size([6, 4, 5])
        assert catted_tc.y.X.shape == torch.Size([6, 4, 5])
        assert (catted_tc.X == 1).all()
        assert (catted_tc.y.X == 1).all()
        assert isinstance(catted_tc._tensordict, TensorDict)
        assert catted_tc.z == catted_tc.y.z == z

        # Testing negative scenarios
        y = torch.zeros(3, 4, 5, dtype=torch.bool)
        data3 = self.MyData(X=X, y=y, z=z, batch_size=batch_size)

        with self.assertRaisesRegex(
            TypeError,
            expected_regex=("Multiple dispatch failed|no implementation found"),
        ):
            torch.cat([data1, data3], dim=0)

    def test_unbind(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        z = "test_tensorclass"
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
        unbind_tcs = torch.unbind(data, 0)
        assert type(unbind_tcs[1]) is type(data)
        assert type(unbind_tcs[0].y[0]) is type(data)
        assert len(unbind_tcs) == 3
        assert torch.all(torch.eq(unbind_tcs[0].X, torch.ones(4, 5)))
        assert torch.all(torch.eq(unbind_tcs[0].y[0].X, torch.ones(4, 5)))
        assert unbind_tcs[0].batch_size == torch.Size([4])
        assert unbind_tcs[0].z == unbind_tcs[1].z == unbind_tcs[2].z == z

    def test_full_like(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        z = "test_tensorclass"
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
        full_like_tc = torch.full_like(data, 9.0)
        assert type(full_like_tc) is type(data)
        assert full_like_tc.batch_size == torch.Size(data.batch_size)
        assert full_like_tc.X.size() == data.X.size()
        assert isinstance(full_like_tc.y, type(data.y))
        assert full_like_tc.y.X.size() == data.y.X.size()
        assert (full_like_tc.X == 9).all()
        assert (full_like_tc.y.X == 9).all()
        assert full_like_tc.z == data.z == z

    def test_clone(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        z = "test_tensorclass"
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
        clone_tc = torch.clone(data)
        assert clone_tc.batch_size == torch.Size(data.batch_size)
        assert torch.all(torch.eq(clone_tc.X, data.X))
        assert isinstance(clone_tc.y, MyDataNested)
        assert torch.all(torch.eq(clone_tc.y.X, data.y.X))
        assert clone_tc.z == data.z == z

    def test_squeeze(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(1, 4, 5)
        z = "test_tensorclass"
        batch_size = [1, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
        squeeze_tc = torch.squeeze(data)
        assert squeeze_tc.batch_size == torch.Size([4])
        assert squeeze_tc.X.shape == torch.Size([4, 5])
        assert squeeze_tc.y.X.shape == torch.Size([4, 5])
        assert squeeze_tc.z == squeeze_tc.y.z == z

    def test_unsqueeze(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        z = "test_tensorclass"
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
        unsqueeze_tc = torch.unsqueeze(data, dim=1)
        assert unsqueeze_tc.batch_size == torch.Size([3, 1, 4])
        assert unsqueeze_tc.X.shape == torch.Size([3, 1, 4, 5])
        assert unsqueeze_tc.y.X.shape == torch.Size([3, 1, 4, 5])
        assert unsqueeze_tc.z == unsqueeze_tc.y.z == z

    def test_split(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(3, 6, 5)
        z = "test_tensorclass"
        batch_size = [3, 6]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data = self.MyData(X=X, y=data_nest, z=z, batch_size=batch_size)
        split_tcs = torch.split(data, split_size_or_sections=[3, 2, 1], dim=1)
        assert type(split_tcs[1]) is type(data)
        assert split_tcs[0].batch_size == torch.Size([3, 3])
        assert split_tcs[1].batch_size == torch.Size([3, 2])
        assert split_tcs[2].batch_size == torch.Size([3, 1])

        assert split_tcs[0].y.batch_size == torch.Size([3, 3])
        assert split_tcs[1].y.batch_size == torch.Size([3, 2])
        assert split_tcs[2].y.batch_size == torch.Size([3, 1])

        assert torch.all(torch.eq(split_tcs[0].X, torch.ones(3, 3, 5)))
        assert torch.all(torch.eq(split_tcs[0].y[0].X, torch.ones(3, 3, 5)))
        assert split_tcs[0].z == split_tcs[1].z == split_tcs[2].z == z
        assert split_tcs[0].y[0].z == split_tcs[0].y[1].z == split_tcs[0].y[2].z == z

    def test_reshape(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        z = "test_tensorclass"
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
        stacked_tc = data.reshape(-1)
        assert stacked_tc.X.shape == torch.Size([12, 5])
        assert stacked_tc.y.X.shape == torch.Size([12, 5])
        assert stacked_tc.shape == torch.Size([12])
        assert (stacked_tc.X == 1).all()
        assert isinstance(stacked_tc._tensordict, TensorDict)
        assert stacked_tc.z == stacked_tc.y.z == z

    def test_view(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        z = "test_tensorclass"
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
        stacked_tc = data.view(-1)
        assert stacked_tc.X.shape == torch.Size([12, 5])
        assert stacked_tc.y.X.shape == torch.Size([12, 5])
        assert stacked_tc.shape == torch.Size([12])
        assert (stacked_tc.X == 1).all()
        assert isinstance(stacked_tc._tensordict, _ViewedTensorDict)
        assert stacked_tc.z == stacked_tc.y.z == z

    def test_permute(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        z = "test_tensorclass"
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
        stacked_tc = data.permute(1, 0)
        assert stacked_tc.X.shape == torch.Size([4, 3, 5])
        assert stacked_tc.y.X.shape == torch.Size([4, 3, 5])
        assert stacked_tc.shape == torch.Size([4, 3])
        assert (stacked_tc.X == 1).all()
        assert isinstance(stacked_tc._tensordict, _PermutedTensorDict)
        assert stacked_tc.z == stacked_tc.y.z == z

    def test_nested(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        z = "test_tensorclass"
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
        assert isinstance(data.y, MyDataNested), type(data.y)
        assert data.z == data_nest.z == data.y.z == z

    def test_nested_eq(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        z = "test_tensorclass"
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
        data_nest2 = MyDataNested(X=X, z=z, batch_size=batch_size)
        data2 = MyDataNested(X=X, y=data_nest2, z=z, batch_size=batch_size)
        assert (data == data2).all()
        assert (data == data2).X.all()
        assert (data == data2).z is None
        assert (data == data2).y.X.all()
        assert (data == data2).y.z is None

    def test_nested_ne(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        z = "test_tensorclass"
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
        data_nest2 = MyDataNested(X=X, z=z, batch_size=batch_size)
        z = "test_bluff"
        data2 = MyDataNested(X=X + 1, y=data_nest2, z=z, batch_size=batch_size)
        assert (data != data2).any()
        assert (data != data2).X.all()
        assert (data != data2).z is None
        assert not (data != data2).y.X.any()
        assert (data != data2).y.z is None

    def test_args(self):
        @tensorclass
        class MyData:
            D: torch.Tensor
            B: torch.Tensor
            A: torch.Tensor
            C: torch.Tensor
            E: str

        D = torch.ones(3, 4, 5)
        B = torch.ones(3, 4, 5)
        A = torch.ones(3, 4, 5)
        C = torch.ones(3, 4, 5)
        E = "test_tensorclass"
        data1 = MyData(D, B=B, A=A, C=C, E=E, batch_size=[3, 4])
        data2 = MyData(D, B, A=A, C=C, E=E, batch_size=[3, 4])
        data3 = MyData(D, B, A, C=C, E=E, batch_size=[3, 4])
        data4 = MyData(D, B, A, C, E=E, batch_size=[3, 4])
        data5 = MyData(D, B, A, C, E, batch_size=[3, 4])
        data = torch.stack([data1, data2, data3, data4, data5], 0)
        assert (data.A == A).all()
        assert (data.B == B).all()
        assert (data.C == C).all()
        assert (data.D == D).all()
        assert data.E == E

    @parametrize("any_to_td", [True, False])
    def test_nested_heterogeneous(self, any_to_td):
        @tensorclass
        class MyDataNest:
            X: torch.Tensor
            v: str

        @tensorclass
        class MyDataParent:
            W: Any
            X: Tensor
            z: TensorDictBase
            y: MyDataNest
            v: str

        batch_size = [3, 4]
        if any_to_td:
            W = TensorDict({}, batch_size)
        else:
            W = torch.zeros(*batch_size, 1)
        X = torch.ones(3, 4, 5)
        data_nest = MyDataNest(X=X, v="test_nested", batch_size=batch_size)
        td = TensorDict({}, batch_size)
        v = "test_tensorclass"
        data = MyDataParent(X=X, y=data_nest, z=td, W=W, v=v, batch_size=batch_size)
        assert isinstance(data.y, MyDataNest)
        assert isinstance(data.y.X, Tensor)
        assert isinstance(data.X, Tensor)
        if not any_to_td:
            assert isinstance(data.W, Tensor)
        else:
            assert isinstance(data.W, TensorDict)
        assert isinstance(data, MyDataParent)
        assert isinstance(data.z, TensorDict)
        assert data.v == v
        assert data.y.v == "test_nested"
        # Testing nested indexing
        assert isinstance(data[0], type(data))
        assert isinstance(data[0].y, type(data.y))
        assert data[0].y.X.shape == torch.Size([4, 5])

    @parametrize("any_to_td", [True, False])
    def test_getattr(self, any_to_td):
        @tensorclass
        class MyDataNest:
            X: torch.Tensor
            v: str

        @tensorclass
        class MyDataParent:
            W: Any
            X: Tensor
            z: TensorDictBase
            y: MyDataNest
            v: str

        batch_size = [3, 4]
        if any_to_td:
            W = TensorDict({}, batch_size)
        else:
            W = torch.zeros(*batch_size, 1)
        X = torch.ones(3, 4, 5)
        td = TensorDict({}, batch_size)
        data_nest = MyDataNest(X=X, v="test_nested", batch_size=batch_size)
        v = "test_tensorclass"
        data = MyDataParent(X=X, y=data_nest, z=td, W=W, v=v, batch_size=batch_size)
        assert isinstance(data.y, type(data_nest))
        assert (data.X == X).all()
        assert data.batch_size == torch.Size(batch_size)
        assert data.v == v
        assert (data.z == td).all()
        assert (data.W == W).all()

        # Testing nested tensor class
        assert data.y._tensordict is data_nest._tensordict
        assert (data.y.X == X).all()
        assert data.y.v == "test_nested"
        assert data.y.batch_size == torch.Size(batch_size)

    @parametrize("any_to_td", [True, False])
    def test_setattr(self, any_to_td):
        @tensorclass
        class MyDataNest:
            X: torch.Tensor
            v: str

        @tensorclass
        class MyDataParent:
            W: Any
            X: Tensor
            z: TensorDictBase
            y: MyDataNest
            v: Any
            k: Optional[Tensor] = None

        batch_size = [3, 4]
        if any_to_td:
            W = TensorDict({}, batch_size)
        else:
            W = torch.zeros(*batch_size, 1)
        X = torch.ones(3, 4, 5)
        td = TensorDict({}, batch_size)
        data_nest = MyDataNest(X=X, v="test_nested", batch_size=batch_size)
        data = MyDataParent(
            X=X, y=data_nest, z=td, W=W, v="test_tensorclass", batch_size=batch_size
        )
        assert isinstance(data.y, type(data_nest))
        assert data.y._tensordict is data_nest._tensordict
        data.X = torch.zeros(3, 4, 5)
        assert (data.X == torch.zeros(3, 4, 5)).all()
        v_new = "test_bluff"
        data.v = v_new
        assert data.v == v_new
        # check that you can't mess up the batch_size
        with self.assertRaisesRegex(
            RuntimeError,
            expected_regex=re.escape("the tensor smth has shape torch.Size([1]) which"),
        ):
            data.z = TensorDict({"smth": torch.zeros(1)}, [])
        # check that you can't write any attribute
        with self.assertRaisesRegex(
            AttributeError, expected_regex=re.escape("Cannot set the attribute")
        ):
            data.newattr = TensorDict({"smth": torch.zeros(1)}, [])
        # Testing nested cases
        data_nest.X = torch.zeros(3, 4, 5)
        assert (data_nest.X == torch.zeros(3, 4, 5)).all()
        assert (data.y.X == torch.zeros(3, 4, 5)).all()
        assert data.y.v == "test_nested"
        data.y.v = "test_nested_new"
        assert data.y.v == data_nest.v == "test_nested_new"
        data_nest.v = "test_nested"
        assert data_nest.v == data.y.v == "test_nested"

        # Testing if user can override the type of the attribute
        data.v = torch.ones(3, 4, 5)
        assert (data.v == torch.ones(3, 4, 5)).all()
        assert "v" in data._tensordict.keys()
        assert "v" not in data._non_tensordict.keys()

        data.v = "test"
        assert data.v == "test"
        assert "v" not in data._tensordict.keys()
        assert "v" in data._non_tensordict.keys()

        # ensure optional fields are writable
        data.k = torch.zeros(3, 4, 5)

    def test_set(self):
        @tensorclass
        class MyDataNest:
            X: torch.Tensor
            v: str

        @tensorclass
        class MyDataParent:
            X: Tensor
            z: TensorDictBase
            y: MyDataNest
            v: str
            k: Optional[Tensor] = None

        batch_size = [3, 4]
        X = torch.ones(3, 4, 5)
        td = TensorDict({}, batch_size)
        data_nest = MyDataNest(X=X, v="test_nested", batch_size=batch_size)
        data = MyDataParent(
            X=X, y=data_nest, z=td, v="test_tensorclass", batch_size=batch_size
        )

        assert isinstance(data.y, type(data_nest))
        assert data.y._tensordict is data_nest._tensordict
        data.set("X", torch.zeros(3, 4, 5))
        assert (data.X == torch.zeros(3, 4, 5)).all()
        v_new = "test_bluff"
        data.set("v", v_new)
        assert data.v == v_new
        # check that you can't mess up the batch_size
        with self.assertRaisesRegex(
            RuntimeError,
            expected_regex=re.escape("the tensor smth has shape torch.Size([1]) which"),
        ):
            data.set("z", TensorDict({"smth": torch.zeros(1)}, []))
        # check that you can't write any attribute
        with self.assertRaisesRegex(
            AttributeError, expected_regex=re.escape("Cannot set the attribute")
        ):
            data.set("newattr", TensorDict({"smth": torch.zeros(1)}, []))

        # Testing nested cases
        data_nest.set("X", torch.zeros(3, 4, 5))
        assert (data_nest.X == torch.zeros(3, 4, 5)).all()
        assert (data.y.X == torch.zeros(3, 4, 5)).all()
        assert data.y.v == "test_nested"
        data.set(("y", "v"), "test_nested_new")
        assert data.y.v == data_nest.v == "test_nested_new"
        data_nest.set("v", "test_nested")
        assert data_nest.v == data.y.v == "test_nested"

        data.set(("y", ("v",)), "this time another string")
        assert data.y.v == data_nest.v == "this time another string"

        # Testing if user can override the type of the attribute
        vorig = torch.ones(3, 4, 5)
        data.set("v", vorig)
        assert (data.v == torch.ones(3, 4, 5)).all()
        assert "v" in data._tensordict.keys()
        assert "v" not in data._non_tensordict.keys()

        data.set("v", torch.zeros(3, 4, 5), inplace=True)
        assert (vorig == 0).all()
        with self.assertRaisesRegex(
            RuntimeError, expected_regex="Cannot update an existing"
        ):
            data.set("v", "les chaussettes", inplace=True)

        data.set("v", "test")
        assert data.v == "test"
        assert "v" not in data._tensordict.keys()
        assert "v" in data._non_tensordict.keys()

        with self.assertRaisesRegex(
            RuntimeError, expected_regex="Cannot update an existing"
        ):
            data.set("v", vorig, inplace=True)

        # ensure optional fields are writable
        data.set("k", torch.zeros(3, 4, 5))

    def test_get(self):
        @tensorclass
        class MyDataNest:
            X: torch.Tensor
            v: str

        @tensorclass
        class MyDataParent:
            X: Tensor
            z: TensorDictBase
            y: MyDataNest
            v: str
            k: Optional[Tensor] = None

        batch_size = [3, 4]
        X = torch.ones(3, 4, 5)
        td = TensorDict({}, batch_size)
        data_nest = MyDataNest(X=X, v="test_nested", batch_size=batch_size)
        v = "test_tensorclass"
        data = MyDataParent(X=X, y=data_nest, z=td, v=v, batch_size=batch_size)
        assert isinstance(data.y, type(data_nest))
        assert (data.get("X") == X).all()
        assert data.get("batch_size") == torch.Size(batch_size)
        assert data.get("v") == v
        assert (data.get("z") == td).all()

        # Testing nested tensor class
        assert data.get("y")._tensordict is data_nest._tensordict
        assert (data.get("y").X == X).all()
        assert (data.get(("y", "X")) == X).all()
        assert data.get("y").v == "test_nested"
        assert data.get(("y", "v")) == "test_nested"
        assert data.get("y").batch_size == torch.Size(batch_size)

        # ensure optional fields are there
        assert data.get("k") is None

        # ensure default works
        assert data.get("foo", "working") == "working"
        assert data.get(("foo", "foo2"), "working") == "working"
        assert data.get(("X", "foo2"), "working") == "working"

        assert (data.get("X", "working") == X).all()
        assert data.get("v", "working") == v

    def test_tensorclass_set_at_(self):
        @tensorclass
        class MyDataNest:
            X: torch.Tensor
            v: str

        @tensorclass
        class MyDataParent:
            X: Tensor
            z: TensorDictBase
            y: MyDataNest
            v: str
            k: Optional[Tensor] = None

        batch_size = [3, 4]
        X = torch.ones(3, 4, 5)
        td = TensorDict({}, batch_size)
        data_nest = MyDataNest(X=X, v="test_nested", batch_size=batch_size)
        v = "test_tensorclass"
        data = MyDataParent(X=X, y=data_nest, z=td, v=v, batch_size=batch_size)

        data.set_at_("X", 5, slice(2, 3))
        data.set_at_(("y", "X"), 5, slice(2, 3))
        assert (data.get_at("X", slice(2, 3)) == 5).all()
        assert (data.get_at(("y", "X"), slice(2, 3)) == 5).all()
        # assert other not changed
        assert (data.get_at("X", slice(0, 2)) == 1).all()
        assert (data.get_at(("y", "X"), slice(0, 2)) == 1).all()
        assert (data.get_at("X", slice(3, 5)) == 1).all()
        assert (data.get_at(("y", "X"), slice(3, 5)) == 1).all()

    def test_tensorclass_get_at(self):
        @tensorclass
        class MyDataNest:
            X: torch.Tensor
            v: str

        @tensorclass
        class MyDataParent:
            X: Tensor
            z: TensorDictBase
            y: MyDataNest
            v: str
            k: Optional[Tensor] = None

        batch_size = [3, 4]
        X = torch.ones(3, 4, 5)
        td = TensorDict({}, batch_size)
        data_nest = MyDataNest(X=X, v="test_nested", batch_size=batch_size)
        v = "test_tensorclass"
        data = MyDataParent(X=X, y=data_nest, z=td, v=v, batch_size=batch_size)

        assert (data.get("X")[2:3] == data.get_at("X", slice(2, 3))).all()
        assert (data.get(("y", "X"))[2:3] == data.get_at(("y", "X"), slice(2, 3))).all()

        # check default
        assert data.get_at(("y", "foo"), slice(2, 3), "working") == "working"
        assert data.get_at("foo", slice(2, 3), "working") == "working"

    def test_pre_allocate(self):
        @tensorclass
        class M1:
            X: Any

        @tensorclass
        class M2:
            X: Any

        @tensorclass
        class M3:
            X: Any

        m1 = M1(M2(M3(X=None, batch_size=[4]), batch_size=[4]), batch_size=[4])
        m2 = M1(M2(M3(X=torch.randn(2), batch_size=[]), batch_size=[]), batch_size=[])
        assert m1.X.X.X is None
        m1[0] = m2
        assert (m1[0].X.X.X == m2.X.X.X).all()

    def test_post_init(self):
        @tensorclass
        class MyDataPostInit:
            X: torch.Tensor
            y: torch.Tensor

            def __post_init__(self):
                assert (self.X > 0).all()
                assert self.y.abs().max() <= 10
                self.y = self.y.abs()

        y = torch.clamp(torch.randn(3, 4), min=-10, max=10)
        data = MyDataPostInit(X=torch.rand(3, 4), y=y, batch_size=[3, 4])
        assert (data.y == y.abs()).all()

        # initialising from tensordict is fine
        data = MyDataPostInit._from_tensordict(
            TensorDict({"X": torch.rand(3, 4), "y": y}, batch_size=[3, 4])
        )

        with self.assertRaises(AssertionError):
            MyDataPostInit(X=-torch.ones(2), y=torch.rand(2), batch_size=[2])

        with self.assertRaises(AssertionError):
            MyDataPostInit._from_tensordict(
                TensorDict({"X": -torch.ones(2), "y": torch.rand(2)}, batch_size=[2])
            )

    def test_default(self):
        @tensorclass
        class MyData:
            X: torch.Tensor = (
                None  # TODO: do we want to allow any default, say an integer?
            )
            y: torch.Tensor = torch.ones(3, 4, 5)

        data = MyData(batch_size=[3, 4])
        assert (data.y == 1).all()
        assert data.X is None
        data.X = torch.zeros(3, 4, 1)
        assert (data.X == 0).all()

        MyData(batch_size=[3])
        MyData(batch_size=[])
        with self.assertRaisesRegex(
            RuntimeError, expected_regex="batch dimension mismatch"
        ):
            MyData(batch_size=[4])

    def test_defaultfactory(self):
        @tensorclass
        class MyData:
            X: torch.Tensor = (
                None  # TODO: do we want to allow any default, say an integer?
            )
            y: torch.Tensor = dataclasses.field(
                default_factory=lambda: torch.ones(3, 4, 5)
            )

        data = MyData(batch_size=[3, 4])
        assert (data.y == 1).all()
        assert data.X is None
        data.X = torch.zeros(3, 4, 1)
        assert (data.X == 0).all()

        MyData(batch_size=[3])
        MyData(batch_size=[])
        with self.assertRaisesRegex(
            RuntimeError, expected_regex="batch dimension mismatch"
        ):
            MyData(batch_size=[4])

    def test_pickle(self):
        data = self.MyData(
            X=torch.ones(3, 4, 5),
            y=torch.zeros(3, 4, 5, dtype=torch.bool),
            z="test_tensorclass",
            batch_size=[3, 4],
        )

        with TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)

            with open(tempdir / "test.pkl", "wb") as f:
                pickle.dump(data, f)

            with open(tempdir / "test.pkl", "rb") as f:
                data2 = pickle.load(f)

        assert_allclose_td(data.to_tensordict(), data2.to_tensordict())
        assert isinstance(data2, self.MyData)
        assert data2.z == data.z

    @classmethod
    def _make_data(cls, shape):
        return cls.MyData(
            X=torch.rand(*shape),
            y=torch.rand(*shape),
            z="test_tensorclass",
            batch_size=shape[:1],
        )

    def test_multiprocessing(self):
        with Pool(os.cpu_count()) as p:
            catted = torch.cat(
                p.map(self._make_data, [(i, 2) for i in range(1, 9)]), dim=0
            )

        assert catted.batch_size == torch.Size([36])
        assert catted.z == "test_tensorclass"

    def test_statedict_errors(self):
        @tensorclass
        class MyClass:
            x: torch.Tensor
            z: str
            y: "MyClass" = None

        z = "test_tensorclass"
        tc = MyClass(
            x=torch.randn(3),
            z=z,
            y=MyClass(x=torch.randn(3), z=z, batch_size=[]),
            batch_size=[],
        )

        sd = tc.state_dict()
        sd["a"] = None
        with self.assertRaisesRegex(
            KeyError, expected_regex="Key 'a' wasn't expected in the state-dict"
        ):
            tc.load_state_dict(sd)
        del sd["a"]
        sd["_tensordict"]["a"] = None
        with self.assertRaisesRegex(
            KeyError, expected_regex="Key 'a' wasn't expected in the state-dict"
        ):
            tc.load_state_dict(sd)
        del sd["_tensordict"]["a"]
        sd["_non_tensordict"]["a"] = None
        with self.assertRaisesRegex(
            KeyError, expected_regex="Key 'a' wasn't expected in the state-dict"
        ):
            tc.load_state_dict(sd)
        del sd["_non_tensordict"]["a"]
        sd["_tensordict"]["y"]["_tensordict"]["a"] = None
        with self.assertRaisesRegex(
            KeyError, expected_regex="Key 'a' wasn't expected in the state-dict"
        ):
            tc.load_state_dict(sd)

    def test_equal(self):
        @tensorclass
        class MyClass1:
            x: torch.Tensor
            z: str
            y: "MyClass1" = None

        @tensorclass
        class MyClass2:
            x: torch.Tensor
            z: str
            y: "MyClass2" = None

        a = MyClass1(
            torch.zeros(3),
            "z0",
            MyClass1(
                torch.ones(3),
                "z1",
                None,
                batch_size=[3],
            ),
            batch_size=[3],
        )
        b = MyClass2(
            torch.zeros(3),
            "z0",
            MyClass2(
                torch.ones(3),
                "z1",
                None,
                batch_size=[3],
            ),
            batch_size=[3],
        )
        c = TensorDict({"x": torch.zeros(3), "y": {"x": torch.ones(3)}}, batch_size=[3])

        assert (a == a.clone()).all()
        assert (a != 1.0).any()
        assert (a[:2] != 1.0).any()

        assert (a.y == 1).all()
        assert (a[:2].y == 1).all()
        assert (a.y[:2] == 1).all()

        assert (a != torch.ones([])).any()
        assert (a.y == torch.ones([])).all()

        assert (a == b).all()
        assert (b == a).all()
        assert (b[:2] == a[:2]).all()

        assert (a == c).all()
        assert (a[:2] == c[:2]).all()

        assert (c == a).all()
        assert (c[:2] == a[:2]).all()

        assert (a != c.clone().zero_()).any()
        assert (c != a.clone().zero_()).any()

    def test_all_any(self):
        @tensorclass
        class MyClass1:
            x: torch.Tensor
            z: str
            y: "MyClass1" = None

        # with all 0
        x = MyClass1(
            torch.zeros(3, 1),
            "z",
            MyClass1(torch.zeros(3, 1), "z", batch_size=[3, 1]),
            batch_size=[3, 1],
        )
        assert not x.all()
        assert not x.any()
        assert isinstance(x.all(), bool)
        assert isinstance(x.any(), bool)
        for dim in [0, 1, -1, -2]:
            assert isinstance(x.all(dim=dim), MyClass1)
            assert isinstance(x.any(dim=dim), MyClass1)
            assert not x.all(dim=dim).all()
            assert not x.any(dim=dim).any()
        # with all 1
        x = x.apply(lambda x: x.fill_(1.0))
        assert isinstance(x, MyClass1)
        assert x.all()
        assert x.any()
        assert isinstance(x.all(), bool)
        assert isinstance(x.any(), bool)
        for dim in [0, 1]:
            assert isinstance(x.all(dim=dim), MyClass1)
            assert isinstance(x.any(dim=dim), MyClass1)
            assert x.all(dim=dim).all()
            assert x.any(dim=dim).any()

        # with 0 and 1
        x.y.x.fill_(0.0)
        assert not x.all()
        assert x.any()
        assert isinstance(x.all(), bool)
        assert isinstance(x.any(), bool)
        for dim in [0, 1]:
            assert isinstance(x.all(dim=dim), MyClass1)
            assert isinstance(x.any(dim=dim), MyClass1)
            assert not x.all(dim=dim).all()
            assert x.any(dim=dim).any()

        assert not x.y.all()
        assert not x.y.any()

    @parametrize("from_torch", [True, False])
    def test_gather(self, from_torch):
        @tensorclass
        class MyClass:
            x: torch.Tensor
            z: str
            y: "MyClass" = None

        c = MyClass(
            torch.randn(3, 4),
            "foo",
            MyClass(torch.randn(3, 4, 5), "bar", None, batch_size=[3, 4, 5]),
            batch_size=[3, 4],
        )
        dim = -1
        index = torch.arange(3).expand(3, 3)
        if from_torch:
            c_gather = torch.gather(c, index=index, dim=dim)
        else:
            c_gather = c.gather(index=index, dim=dim)
        assert c_gather.x.shape == torch.Size([3, 3])
        assert c_gather.y.shape == torch.Size([3, 3, 5])
        assert c_gather.y.x.shape == torch.Size([3, 3, 5])
        assert c_gather.y.z == "bar"
        assert c_gather.z == "foo"
        c_gather_zero = c_gather.clone().zero_()
        if from_torch:
            c_gather2 = torch.gather(c, index=index, dim=dim, out=c_gather_zero)
        else:
            c_gather2 = c.gather(index=index, dim=dim, out=c_gather_zero)

        assert (c_gather2 == c_gather).all()

    def test_to_tensordict(self):
        @tensorclass
        class MyClass:
            x: torch.Tensor
            z: str
            y: "MyClass" = None

        c = MyClass(
            torch.randn(3, 4),
            "foo",
            MyClass(torch.randn(3, 4, 5), "bar", None, batch_size=[3, 4, 5]),
            batch_size=[3, 4],
        )

        ctd = c.to_tensordict()
        assert isinstance(ctd, TensorDictBase)
        assert "x" in ctd.keys()
        assert "z" not in ctd.keys()
        assert "y" in ctd.keys()
        assert ("y", "x") in ctd.keys(True)

    def test_memmap_(self):
        @tensorclass
        class MyClass:
            x: torch.Tensor
            z: str
            y: "MyClass" = None

        c = MyClass(
            torch.randn(3, 4),
            "foo",
            MyClass(torch.randn(3, 4, 5), "bar", None, batch_size=[3, 4, 5]),
            batch_size=[3, 4],
        )

        cmemmap = c.memmap_()
        assert cmemmap is c
        assert isinstance(c.x, MemoryMappedTensor)
        assert isinstance(c.y.x, MemoryMappedTensor)
        assert c.z == "foo"

    def test_memmap_like(self):
        @tensorclass
        class MyClass:
            x: torch.Tensor
            z: str
            y: "MyClass" = None

        c = MyClass(
            torch.randn(3, 4),
            "foo",
            MyClass(torch.randn(3, 4, 5), "bar", None, batch_size=[3, 4, 5]),
            batch_size=[3, 4],
        )

        cmemmap = c.memmap_like()
        assert cmemmap is not c
        assert cmemmap.y is not c.y
        assert (cmemmap == 0).all()
        assert isinstance(cmemmap.x, MemoryMappedTensor)
        assert isinstance(cmemmap.y.x, MemoryMappedTensor)
        assert cmemmap.z == "foo"

    def test_from_memmap(self):
        with TemporaryDirectoryName() as tmpdir:
            td = TensorDict(
                {
                    ("a", "b", "c"): 1,
                    ("a", "d"): 2,
                },
                [],
            ).expand(10)
            td.memmap_(tmpdir)

            @tensorclass
            class MyClass:
                a: TensorDictBase

            tc = MyClass.load_memmap(tmpdir)
            assert isinstance(tc.a, TensorDict)
            assert tc.batch_size == torch.Size([10])

    def test_from_dict(self):
        td = TensorDict(
            {
                ("a", "b", "c"): 1,
                ("a", "d"): 2,
            },
            [],
        ).expand(10)
        d = td.to_dict()

        @tensorclass
        class MyClass:
            a: TensorDictBase

        tc = MyClass.from_dict(d)
        assert isinstance(tc, MyClass)
        assert isinstance(tc.a, TensorDict)
        assert tc.batch_size == torch.Size([10])


instantiate_parametrized_tests(TestTensorClass)


class TestNesting:
    @tensorclass
    class TensorClass:
        tens: torch.Tensor
        order: Tuple[str]
        test: str

    def get_nested(self):
        c = self.TensorClass(torch.ones(1), ("a", "b", "c"), "Hello", batch_size=[])

        td = torch.stack(
            [TensorDict({"t": torch.ones(1), "c": c}, batch_size=[]) for _ in range(3)]
        )
        return td

    def test_to(self):
        td = self.get_nested()
        td = td.to("cpu:1")
        assert isinstance(td.get("c")[0], self.TensorClass)

    def test_idx(self):
        td = self.get_nested()[0]
        assert isinstance(td.get("c"), self.TensorClass)

    def test_apply(self):
        td = self.get_nested()
        td = td.apply(lambda x: x + 1)
        assert isinstance(td.get("c")[0], self.TensorClass)

    def test_split(self):
        td = self.get_nested()
        td, _ = td.split([2, 1], dim=0)
        assert isinstance(td.get("c")[0], self.TensorClass)

    def test_chunk(self):
        td = self.get_nested()
        td, _ = td.chunk(2, dim=0)
        assert isinstance(td.get("c")[0], self.TensorClass)


if __name__ == "__main__":
    run_tests()

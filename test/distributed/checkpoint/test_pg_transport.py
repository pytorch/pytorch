# Owner(s): ["oncall: distributed"]

import logging
import os
import sys
import unittest
from datetime import timedelta
from typing import Optional
from unittest import skipIf, skipUnless
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist

import torch.nn as nn
from torch.distributed import TCPStore
from torch.distributed.checkpoint._pg_transport import (
    _cast_tensor,
    _DTensorMeta,
    _prepare_state_dict,
    _prepare_tensor,
    _StateDictMeta,
    _TensorMeta,
    _timeit,
    PGTransport,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.distributed_c10d import _get_default_group

from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import (
    MultiProcContinousTest,
    requires_nccl,
)

from torch.distributed.tensor import DTensor

from torch.testing._internal.common_utils import (
    check_leaked_tensors,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TestCase,
)


logger = logging.getLogger(__name__)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 10)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def ring_send_recv_checkpoint(
    transport: PGTransport, state_dict, rank, world_size, step=0
):
    """
    Use the transport to send to rank + 1 and receive from rank - 1.
    """
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size
    if rank == 0:
        transport.send_checkpoint([next_rank], state_dict)
        received_checkpoint = transport.recv_checkpoint(prev_rank)
    else:
        received_checkpoint = transport.recv_checkpoint(prev_rank)
        transport.send_checkpoint([next_rank], received_checkpoint)
    return received_checkpoint


def _test_pg_transport(self, device) -> None:
    # python test/distributed/checkpoint/test_pg_transport.py -k test_pg_transport
    print(f"{self.rank=} pid: {os.getpid()} {device=}")
    print("in test")

    model = SimpleModel().to(device)
    transport = PGTransport(_get_default_group(), timedelta(seconds=10), device)
    original_state_dict = model.state_dict()
    received_checkpoint = ring_send_recv_checkpoint(
        transport=transport,
        state_dict=original_state_dict,
        rank=self.rank,
        world_size=self.world_size,
    )
    self.assertEqual(original_state_dict, received_checkpoint)


def _test_pg_transport_with_dtensor(self, device) -> None:
    # Create a device mesh for DTensor
    device_mesh = init_device_mesh(device.type, (self.world_size,))

    # Create a DTensor
    local_tensor = torch.randn(10, 10, device=device)
    dtensor = DTensor.from_local(local_tensor, device_mesh)

    # Include DTensor in the state dict
    model = SimpleModel().to(device)
    state_dict = model.state_dict()
    state_dict["dtensor"] = dtensor

    transport = PGTransport(_get_default_group(), timedelta(seconds=10), device)
    received_checkpoint = ring_send_recv_checkpoint(
        transport=transport,
        state_dict=state_dict,
        rank=self.rank,
        world_size=self.world_size,
    )
    self.assertEqual(state_dict, received_checkpoint)


class PgTransportCPU(MultiProcContinousTest):
    world_size = 8
    timeout: timedelta = timedelta(seconds=20)

    @classmethod
    def backend_str(cls) -> Optional[str]:
        return "gloo"

    @classmethod
    def device_type(cls) -> str:
        return "cpu"

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_type())

    def test_pg_transport(self) -> None:
        _test_pg_transport(self, self.device)

    def test_pg_transport_with_dtensor(self) -> None:
        _test_pg_transport_with_dtensor(self, self.device)


@skipIf(not torch.cuda.is_available(), "CUDA not available")
class PgTransportCUDA(MultiProcContinousTest):
    world_size = 2
    timeout: timedelta = timedelta(seconds=20)

    @classmethod
    def backend_str(cls) -> Optional[str]:
        return "nccl"

    @classmethod
    def device_type(cls) -> str:
        return "cuda"

    @property
    def device(self) -> torch.device:
        return torch.device(f"{self.device_type()}:{self.rank}")

    def test_pg_transport(self) -> None:
        _test_pg_transport(self, self.device)

    def test_pg_transport_with_dtensor(self) -> None:
        _test_pg_transport_with_dtensor(self, self.device)


class TestCastTensor(TestCase):
    def test_cast_tensor_different_dtypes(self):
        """Test casting tensors of different dtypes."""
        dtypes = [torch.float32, torch.float64, torch.int32, torch.int64, torch.bool]

        for dtype in dtypes:
            original = torch.tensor([1, 2, 3], dtype=dtype)
            casted = _cast_tensor(original, torch.uint8)

            # Check that the storage is the same
            self.assertIs(original.untyped_storage(), casted.untyped_storage())

            # Check that the size is correct
            self.assertEqual(casted.numel(), original.untyped_storage().nbytes())

    def test_cast_tensor_with_stride(self):
        """Test casting tensors with non-standard strides."""
        # Create a tensor with non-standard stride
        original = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        transposed = original.t()  # Transpose to get non-standard stride

        casted = _cast_tensor(transposed, torch.uint8)

        # Check that the storage is the same
        self.assertIs(transposed.untyped_storage(), casted.untyped_storage())

        # Check that the size is correct
        self.assertEqual(casted.numel(), transposed.untyped_storage().nbytes())

    def test_cast_tensor_with_offset(self):
        """Test casting tensors with storage offset."""
        # Create a tensor with storage offset
        original = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        sliced = original[2:]  # This creates a tensor with storage offset

        casted = _cast_tensor(sliced, torch.uint8)

        # Check that the storage is the same
        self.assertIs(sliced.untyped_storage(), casted.untyped_storage())

        # Check that the size is correct
        self.assertEqual(casted.numel(), sliced.untyped_storage().nbytes())


class TestPrepareTensor(TestCase):
    def test_prepare_tensor_basic(self):
        """Test basic tensor preparation."""
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        prepared_tensor, meta = _prepare_tensor(tensor)

        # Check metadata
        self.assertEqual(meta.shape, tensor.shape)
        self.assertEqual(meta.dtype, tensor.dtype)
        self.assertEqual(meta.storage_offset, tensor.storage_offset())
        self.assertEqual(meta.stride, tensor.stride())
        self.assertEqual(meta.nbytes, tensor.untyped_storage().nbytes())

        # Check prepared tensor
        self.assertEqual(prepared_tensor.dtype, torch.uint8)
        self.assertEqual(prepared_tensor.numel(), tensor.untyped_storage().nbytes())

    def test_prepare_tensor_different_shapes(self):
        """Test preparing tensors with different shapes."""
        shapes = [(3,), (2, 3), (2, 3, 4)]

        for shape in shapes:
            tensor = torch.randn(shape)
            prepared_tensor, meta = _prepare_tensor(tensor)

            # Check metadata
            self.assertEqual(meta.shape, tensor.shape)
            self.assertEqual(meta.dtype, tensor.dtype)
            self.assertEqual(meta.storage_offset, tensor.storage_offset())
            self.assertEqual(meta.stride, tensor.stride())
            self.assertEqual(meta.nbytes, tensor.untyped_storage().nbytes())

    def test_prepare_tensor_with_stride(self):
        """Test preparing tensors with non-standard strides."""
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        transposed = tensor.t()  # Transpose to get non-standard stride

        prepared_tensor, meta = _prepare_tensor(transposed)

        # Check metadata
        self.assertEqual(meta.shape, transposed.shape)
        self.assertEqual(meta.dtype, transposed.dtype)
        self.assertEqual(meta.storage_offset, transposed.storage_offset())
        self.assertEqual(meta.stride, transposed.stride())
        self.assertEqual(meta.nbytes, transposed.untyped_storage().nbytes())


class TestPrepareStateDict(TestCase):
    def test_prepare_state_dict_basic(self):
        """Test basic state dict preparation."""
        state_dict = {"weight": torch.randn(3, 4), "bias": torch.randn(4)}
        device = torch.device("cpu")

        meta, tensors = _prepare_state_dict(state_dict, device)

        # Check metadata
        self.assertEqual(len(meta.paths), 2)
        self.assertEqual(len(meta.non_tensor_leaves), 2)
        self.assertEqual(len(tensors), 2)

        # Check that all non_tensor_leaves are _TensorMeta instances
        for leaf in meta.non_tensor_leaves:
            self.assertIsInstance(leaf, _TensorMeta)

    def test_prepare_state_dict_nested(self):
        """Test preparing nested state dict."""
        state_dict = {
            "layer1": {"weight": torch.randn(3, 4), "bias": torch.randn(4)},
            "layer2": {"weight": torch.randn(4, 5), "bias": torch.randn(5)},
        }
        device = torch.device("cpu")

        meta, tensors = _prepare_state_dict(state_dict, device)

        # Check metadata
        self.assertEqual(len(meta.paths), 4)
        self.assertEqual(len(meta.non_tensor_leaves), 4)
        self.assertEqual(len(tensors), 4)

    def test_prepare_state_dict_with_non_tensor_values(self):
        """Test preparing state dict with non-tensor values."""
        state_dict = {
            "weight": torch.randn(3, 4),
            "bias": torch.randn(4),
            "config": {"lr": 0.01, "momentum": 0.9},
            "step": 42,
        }
        device = torch.device("cpu")

        meta, tensors = _prepare_state_dict(state_dict, device)

        # Check metadata - the actual number of paths depends on how the pytree flattens the dict
        # The nested config dict might be flattened differently
        self.assertEqual(len(meta.non_tensor_leaves), len(meta.paths))
        self.assertEqual(len(tensors), 2)

        # Check that non-tensor values are preserved
        non_tensor_values = [
            leaf for leaf in meta.non_tensor_leaves if not isinstance(leaf, _TensorMeta)
        ]
        self.assertEqual(len(non_tensor_values), 3)  # config (2) and step


class TestPGTransportMocked(TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.pg = MagicMock()
        self.timeout = timedelta(seconds=10)

        # Mock Work object
        self.mock_work = MagicMock()
        self.mock_work.wait = MagicMock()

        # Setup process group mock to return mock_work
        self.pg.send = MagicMock(return_value=self.mock_work)
        self.pg.recv = MagicMock(return_value=self.mock_work)

    def test_send_checkpoint_basic(self):
        """Test basic send_checkpoint functionality with mocked process group."""
        transport = PGTransport(self.pg, self.timeout, self.device)
        state_dict = {"weight": torch.randn(3, 4), "bias": torch.randn(4)}
        dst_ranks = [1, 2]

        transport.send_checkpoint(dst_ranks, state_dict)

        # Check that send was called with correct parameters
        # First for metadata length, then for metadata, then for each tensor
        expected_calls = len(dst_ranks) * (2 + len(state_dict))
        self.assertEqual(self.pg.send.call_count, expected_calls)

        # Check that wait was called on all work objects
        self.assertEqual(self.mock_work.wait.call_count, expected_calls)

    def test_recv_checkpoint_basic(self):
        """Test basic recv_checkpoint functionality with mocked process group."""
        # Setup mock for pickle.loads to return a valid _StateDictMeta
        with patch("pickle.loads") as mock_loads:
            # Create a mock state dict metadata
            from torch.utils._pytree import tree_flatten_with_path, tree_unflatten

            state_dict = {"weight": torch.randn(3, 4), "bias": torch.randn(4)}
            leaves, treespec = tree_flatten_with_path(state_dict)
            paths = [path for path, _ in leaves]

            # Create mock tensor metadata
            tensor_metas = []
            for _, v in leaves:
                tensor_metas.append(
                    _TensorMeta(
                        shape=v.shape,
                        dtype=v.dtype,
                        storage_offset=v.storage_offset(),
                        stride=v.stride(),
                        nbytes=v.untyped_storage().nbytes(),
                    )
                )

            mock_meta = _StateDictMeta(
                treespec=treespec, paths=paths, non_tensor_leaves=tensor_metas
            )
            mock_loads.return_value = mock_meta

            # Setup len_t and buf tensors for the mock recv
            def side_effect(tensor_list, *args, **kwargs):
                if tensor_list[0].numel() == 1:  # This is len_t
                    tensor_list[0].fill_(100)  # Some arbitrary length
                return self.mock_work

            self.pg.recv.side_effect = side_effect

            # Create transport and call recv_checkpoint
            transport = PGTransport(self.pg, self.timeout, self.device)
            result = transport.recv_checkpoint(src_rank=0)

            # Check that recv was called
            self.assertGreaterEqual(
                self.pg.recv.call_count, 2
            )  # At least for len_t and buf

            # Check that wait was called
            self.assertGreaterEqual(self.mock_work.wait.call_count, 2)

    def test_send_checkpoint_empty_state_dict(self):
        """Test send_checkpoint with empty state dict."""
        transport = PGTransport(self.pg, self.timeout, self.device)
        state_dict = {}
        dst_ranks = [1]

        transport.send_checkpoint(dst_ranks, state_dict)

        # Check that send was called only for metadata
        self.assertEqual(self.pg.send.call_count, 2)  # len_t and buf_t

        # Check that wait was called
        self.assertEqual(self.mock_work.wait.call_count, 2)

    def test_send_checkpoint_with_non_tensor_values(self):
        """Test send_checkpoint with non-tensor values in state dict."""
        transport = PGTransport(self.pg, self.timeout, self.device)
        state_dict = {"weight": torch.randn(3, 4), "config": {"lr": 0.01}}
        dst_ranks = [1]

        transport.send_checkpoint(dst_ranks, state_dict)

        # Check that send was called for metadata and one tensor
        self.assertEqual(self.pg.send.call_count, 3)  # len_t, buf_t, and one tensor

        # Check that wait was called
        self.assertEqual(self.mock_work.wait.call_count, 3)

    def test_recv_checkpoint_with_state_dict_callback(self):
        """Test recv_checkpoint with state_dict callback."""
        # Setup mock for pickle.loads to return a valid _StateDictMeta
        with patch("pickle.loads") as mock_loads:
            # Create a mock state dict metadata
            from torch.utils._pytree import tree_flatten_with_path, tree_unflatten

            state_dict = {"weight": torch.randn(3, 4), "bias": torch.randn(4)}
            leaves, treespec = tree_flatten_with_path(state_dict)
            paths = [path for path, _ in leaves]

            # Create mock tensor metadata
            tensor_metas = []
            for _, v in leaves:
                tensor_metas.append(
                    _TensorMeta(
                        shape=v.shape,
                        dtype=v.dtype,
                        storage_offset=v.storage_offset(),
                        stride=v.stride(),
                        nbytes=v.untyped_storage().nbytes(),
                    )
                )

            mock_meta = _StateDictMeta(
                treespec=treespec, paths=paths, non_tensor_leaves=tensor_metas
            )
            mock_loads.return_value = mock_meta

            # Setup len_t and buf tensors for the mock recv
            def side_effect(tensor_list, *args, **kwargs):
                if tensor_list[0].numel() == 1:  # This is len_t
                    tensor_list[0].fill_(100)  # Some arbitrary length
                return self.mock_work

            self.pg.recv.side_effect = side_effect

            # Create a state_dict callback
            callback_state_dict = {"weight": torch.randn(3, 4), "bias": torch.randn(4)}
            state_dict_callback = MagicMock(return_value=callback_state_dict)

            # Create transport with state_dict callback and call recv_checkpoint
            transport = PGTransport(
                self.pg, self.timeout, self.device, state_dict=state_dict_callback
            )
            result = transport.recv_checkpoint(src_rank=0)

            # Check that state_dict callback was called
            state_dict_callback.assert_called_once()


class TestPGTransportEdgeCases(TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.pg = MagicMock()
        self.timeout = timedelta(seconds=10)

        # Mock Work object
        self.mock_work = MagicMock()
        self.mock_work.wait = MagicMock()

        # Setup process group mock to return mock_work
        self.pg.send = MagicMock(return_value=self.mock_work)
        self.pg.recv = MagicMock(return_value=self.mock_work)

    def test_send_checkpoint_with_mixed_content(self):
        # TODO skipping
        return
        """Test send_checkpoint with a mix of tensors, DTensors, and non-tensor values."""

        # Create a proper mock DTensor class
        class MockDTensor:
            def __init__(self, local_tensor=None, spec=None):
                self._local_tensor = (
                    local_tensor if local_tensor is not None else torch.randn(3, 4)
                )
                self._spec = spec

        # Store the original DTensor class
        original_DTensor = None
        if hasattr(torch.distributed.checkpoint._pg_transport, "DTensor"):
            original_DTensor = torch.distributed.checkpoint._pg_transport.DTensor

        # Replace with our mock class
        torch.distributed.checkpoint._pg_transport.DTensor = MockDTensor

        # Create a mock DTensor instance
        mock_dtensor_spec = MagicMock()
        mock_dtensor = MockDTensor(torch.randn(3, 4), mock_dtensor_spec)

        # Create a state dict with mixed content
        state_dict = {
            "tensor": torch.randn(2, 3),
            "dtensor": mock_dtensor,
            "config": {"lr": 0.01},
            "nested": {"tensor": torch.randn(1, 2), "value": 42},
        }
        with patch(
            "torch.distributed.checkpoint._pg_transport._prepare_tensor"
        ) as mock_prepare_tensor:
            # Setup mock for _prepare_tensor
            mock_tensor = torch.randn(3, 4)
            mock_meta = MagicMock()
            mock_prepare_tensor.return_value = (mock_tensor, mock_meta)

            # Create transport and call send_checkpoint
            transport = PGTransport(self.pg, self.timeout, self.device)
            transport.send_checkpoint([1], state_dict)

            # Check that _prepare_tensor was called for each tensor
            self.assertEqual(mock_prepare_tensor.call_count, 3)  # 2 tensors + 1 dtensor

    def test_send_checkpoint_with_cpu_tensors(self):
        """Test send_checkpoint with CPU tensors when device is CUDA."""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        device = torch.device("cuda:0")

        # Create a state dict with CPU tensors
        state_dict = {
            "cpu_tensor1": torch.randn(2, 3),
            "cpu_tensor2": torch.randn(3, 4),
        }

        # Create transport with CUDA device
        transport = PGTransport(self.pg, self.timeout, device)

        # Call send_checkpoint
        transport.send_checkpoint([1], state_dict)

        # Check that send was called
        self.assertGreaterEqual(
            self.pg.send.call_count, 4
        )  # len_t, buf_t, and 2 tensors

        # Check that wait was called
        self.assertGreaterEqual(self.mock_work.wait.call_count, 4)


# import fbvscode
# fbvscode.attach_debugger()

if __name__ == "__main__":
    run_tests()

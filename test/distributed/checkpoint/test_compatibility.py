# Owner(s): ["oncall: distributed"]

from unittest.mock import patch

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
    _MEM_FORMAT_ENCODING,
)
from torch.distributed._shard.sharded_tensor.metadata import (
    MEM_FORMAT_ENCODING as SHARDED_MEM_FORMAT_ENCODING,
    TensorProperties as ShardedTensorProperties,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


class TestDCPCompatbility(TestCase):
    def test_metadata(self) -> None:
        # Ensure that all the new fields of all the metadata have the default
        # values so that we can always deserialize from a legacy metadata.
        try:
            tensor = torch.zeros(4, 4)
            chunk_meta = ChunkStorageMetadata(
                torch.Size((1, 1)),
                torch.Size((1, 1)),
            )
            tensor_meta = TensorStorageMetadata(
                properties=TensorProperties.create_from_tensor(tensor),
                size=tensor.size(),
                chunks=[chunk_meta],
            )
            b_meta = BytesStorageMetadata()
            _ = Metadata(state_dict_metadata={"a": tensor_meta, "b": b_meta})

            _ = MetadataIndex(fqn="a.b.c")
        except Exception as e:
            raise RuntimeError(
                "The change may break the BC of distributed checkpoint."
            ) from e

    def test_sharded_tensor_dependency(self) -> None:
        # Ensure that we can load the existing DCP checkpoints back even if the
        # metadata contain # _shard.sharded_tensor.metadata.
        from torch.distributed._shard.sharded_tensor.metadata import (
            TensorProperties as stp,
        )

        with patch("torch.distributed.checkpoint.metadata.TensorProperties", stp):
            dcp.save(
                {"a": torch.zeros(4, 4)},
                dcp.FileSystemWriter("/tmp/dcp_testing"),
            )

        dcp.load(
            {"a": torch.zeros(4, 4)},
            dcp.FileSystemReader("/tmp/dcp_testing"),
        )

    @with_temp_dir
    def test_storage_meta(self) -> None:
        writer = dcp.FileSystemWriter(self.temp_dir)
        dcp.save({"a": torch.zeros(4, 4)}, storage_writer=writer)

        reader = dcp.FileSystemReader(self.temp_dir)
        storage_meta = reader.read_metadata().storage_meta
        self.assertNotEqual(storage_meta, None)
        self.assertEqual(str(storage_meta.checkpoint_id), self.temp_dir)
        self.assertEqual(storage_meta.save_id, writer.save_id)
        self.assertEqual(storage_meta.load_id, reader.load_id)

    @with_temp_dir
    def test_with_v_2_3(self) -> None:
        sd = {
            "a": torch.zeros(4, 4),
            "dict": {
                "dict_a": {"dict_a_1": 1, "dict_a_2": 2},
                "dict_b": {"dict_b_1": 1, "dict_b_2": 2},
            },
            "list": [0, 1, 2, 3, 4, 5],
        }
        load_sd = {
            "a": torch.ones(4, 4),
            "dict": {
                "dict_a": {"dict_a_1": 2, "dict_a_2": 4},
                "dict_b": {"dict_b_1": 2, "dict_b_2": 4},
            },
            "list": [10, 11, 12, 13, 14, 15],
        }

        dcp._version._act_like_version = "2_3"
        dcp.save(sd, checkpoint_id=self.temp_dir)
        dcp._version._act_like_version = None
        dcp.load(load_sd, checkpoint_id=self.temp_dir)
        self.assertEqual(sd, load_sd)

    def test_tensor_properties_backward_compat_without_strides(self) -> None:
        """Ensure TensorProperties pickled before the strides field was added can still be loaded."""
        import pickle

        # Simulate an old checkpoint's pickled TensorProperties (5-element state, no strides).
        props = TensorProperties(dtype=torch.float32)
        old_state = (
            torch.float32,
            torch.strided,
            False,
            _MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT,
            False,
        )

        # Verify the current round-trip works (6-element state).
        props_bytes = pickle.dumps(props)
        restored_new = pickle.loads(props_bytes)
        self.assertIsNone(restored_new.strides)

        # Directly test __setstate__ with a 5-element tuple.
        restored_old = TensorProperties.__new__(TensorProperties)
        restored_old.__setstate__(old_state)
        self.assertIsNone(restored_old.strides)
        self.assertEqual(restored_old.dtype, torch.float32)
        self.assertEqual(restored_old.memory_format, torch.contiguous_format)

    def test_tensor_properties_strides_survive_pickle_roundtrip(self) -> None:
        """Ensure strides survive a pickle round-trip (as used by torch.save/load)."""
        import pickle

        tensor = torch.rand(3, 4)
        props = TensorProperties.create_from_tensor(tensor)
        self.assertEqual(props.strides, (4, 1))

        restored = pickle.loads(pickle.dumps(props))
        self.assertEqual(restored.strides, (4, 1))
        self.assertEqual(restored.dtype, torch.float32)

    def test_sharded_tensor_properties_backward_compat_without_strides(self) -> None:
        """Ensure sharded tensor TensorProperties also handles old pickles without strides."""
        old_state = (
            torch.float32,
            torch.strided,
            False,
            SHARDED_MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT,
            False,
        )
        restored = ShardedTensorProperties.__new__(ShardedTensorProperties)
        restored.__setstate__(old_state)
        self.assertIsNone(restored.strides)
        self.assertEqual(restored.dtype, torch.float32)

    def test_sharded_tensor_properties_strides_roundtrip(self) -> None:
        """Ensure sharded tensor TensorProperties preserves strides through getstate/setstate."""
        tensor = torch.rand(3, 4)
        props = ShardedTensorProperties.create_from_tensor(tensor)
        self.assertEqual(props.strides, (4, 1))

        state = props.__getstate__()
        self.assertEqual(len(state), 6)

        restored = ShardedTensorProperties.__new__(ShardedTensorProperties)
        restored.__setstate__(state)
        self.assertEqual(restored.strides, (4, 1))
        self.assertEqual(restored.dtype, torch.float32)

    def test_sharded_tensor_properties_non_contiguous_strides(self) -> None:
        """Ensure sharded tensor TensorProperties captures non-contiguous strides."""
        tensor = torch.rand(5, 10).t()  # shape [10, 5], strides (1, 10)
        props = ShardedTensorProperties.create_from_tensor(tensor)
        self.assertEqual(props.strides, (1, 10))

        state = props.__getstate__()
        restored = ShardedTensorProperties.__new__(ShardedTensorProperties)
        restored.__setstate__(state)
        self.assertEqual(restored.strides, (1, 10))


if __name__ == "__main__":
    run_tests()

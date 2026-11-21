"""
Regression test for DCP metadata restricted unpickler.

Before the fix, FileSystemReader.read_metadata() used raw pickle.load()
to deserialize the .metadata sidecar, while tensor payloads on the SAME
load path already used weights_only=True. This asymmetry meant a crafted
.metadata file could execute arbitrary code even when weights_only=True
was specified (CVE-2025-32434 class).

The fix replaces raw pickle.load() with a restricted unpickler that only
allows known-safe DCP metadata types, with an extension hook
(add_safe_metadata_globals) for plugin authors.
"""

import io
import pickle
import unittest

import torch
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    StorageMeta,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.filesystem import (
    _restricted_metadata_load,
    _RestrictedMetadataUnpickler,
    add_safe_metadata_globals,
)


class TestRestrictedMetadataUnpickler(unittest.TestCase):
    """Verify the restricted unpickler allows safe types and blocks unsafe."""

    def _roundtrip(self, obj):
        """Pickle an object and load it through the restricted unpickler."""
        buf = io.BytesIO()
        pickle.dump(obj, buf)
        buf.seek(0)
        return _restricted_metadata_load(buf)

    def test_normal_metadata_loads(self):
        """Standard DCP Metadata should load without error."""
        metadata = Metadata(
            state_dict_metadata={
                "layer.weight": TensorStorageMetadata(
                    properties=TensorProperties(
                        dtype=torch.float32,
                        layout=torch.strided,
                        requires_grad=True,
                    ),
                    size=torch.Size([128, 64]),
                    chunks=[
                        ChunkStorageMetadata(
                            properties=TensorProperties(
                                dtype=torch.float32,
                            ),
                            size=torch.Size([128, 64]),
                        )
                    ],
                ),
            },
            storage_meta=StorageMeta(checkpoint_id="test-ckpt"),
            version="1.0.0",
        )
        loaded = self._roundtrip(metadata)
        self.assertIsInstance(loaded, Metadata)
        self.assertIn("layer.weight", loaded.state_dict_metadata)
        self.assertEqual(loaded.version, "1.0.0")
        self.assertEqual(loaded.storage_meta.checkpoint_id, "test-ckpt")

    def test_malicious_payload_blocked(self):
        """Non-allowlisted types must raise UnpicklingError."""
        # Craft a pickle payload that tries to instantiate os.system
        import os

        class _MaliciousPayload:
            def __reduce__(self):
                return (os.system, ("echo pwned",))

        buf = io.BytesIO()
        pickle.dump(_MaliciousPayload(), buf)
        buf.seek(0)

        with self.assertRaises(pickle.UnpicklingError) as ctx:
            _restricted_metadata_load(buf)
        self.assertIn("Blocked unpickling", str(ctx.exception))

    def test_metadata_index_loads(self):
        """MetadataIndex should load correctly."""
        idx = MetadataIndex(fqn="model.layer1.weight", storage_index=(0, 0))
        loaded = self._roundtrip(idx)
        self.assertIsInstance(loaded, MetadataIndex)
        self.assertEqual(loaded.fqn, "model.layer1.weight")

    def test_add_safe_metadata_globals_with_mapping(self):
        """Users can register custom types via add_safe_metadata_globals."""

        class CustomData:
            def __init__(self, value=42):
                self.value = value

        # Register the custom type
        add_safe_metadata_globals(
            {CustomData.__module__: {CustomData.__qualname__}}
        )

        obj = CustomData(value=99)
        loaded = self._roundtrip(obj)
        self.assertIsInstance(loaded, CustomData)
        self.assertEqual(loaded.value, 99)

    def test_add_safe_metadata_globals_with_list(self):
        """add_safe_metadata_globals also accepts a list of callables."""

        class AnotherCustom:
            def __init__(self, name="test"):
                self.name = name

        add_safe_metadata_globals([AnotherCustom])

        obj = AnotherCustom(name="hello")
        loaded = self._roundtrip(obj)
        self.assertIsInstance(loaded, AnotherCustom)
        self.assertEqual(loaded.name, "hello")


if __name__ == "__main__":
    unittest.main()

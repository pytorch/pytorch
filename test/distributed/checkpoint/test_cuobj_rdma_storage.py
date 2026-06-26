# Owner(s): ["oncall: distributed checkpointing"]

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint._cuobj_rdma_storage import (
    _parse_s3_url,
    ObjectClient,
    S3RdmaStorageReader,
    S3RdmaStorageWriter,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class _InMemoryClient(ObjectClient):
    """Loopback ObjectClient that stands in for the cuObject/S3 RDMA transport.

    It lets the StorageWriter/StorageReader logic be exercised end to end with
    no RDMA hardware, S3 endpoint, or boto3 dependency.
    """

    def __init__(self) -> None:
        self.objects: dict[str, object] = {}

    def put_rdma(self, key: str, tensor: torch.Tensor) -> None:
        self.objects[key] = tensor.detach().cpu().contiguous().clone()

    def get_rdma(self, key: str, tensor: torch.Tensor) -> None:
        tensor.copy_(self.objects[key])

    def put_bytes(self, key: str, data: bytes) -> None:
        self.objects[key] = bytes(data)

    def get_bytes(self, key: str) -> bytes:
        return self.objects[key]

    def exists(self, key: str) -> bool:
        return key in self.objects


class TestCuObjRdmaStorage(TestCase):
    def test_parse_s3_url(self) -> None:
        self.assertEqual(_parse_s3_url("s3://bucket/a/b"), ("bucket", "a/b"))
        self.assertEqual(_parse_s3_url("s3://bucket"), ("bucket", ""))
        with self.assertRaises(ValueError):
            _parse_s3_url("/local/path")

    def test_validate_checkpoint_id(self) -> None:
        self.assertTrue(S3RdmaStorageWriter.validate_checkpoint_id("s3://b/k"))
        self.assertFalse(S3RdmaStorageWriter.validate_checkpoint_id("/local/k"))

    def test_save_load_roundtrip(self) -> None:
        client = _InMemoryClient()
        state = {
            "weight": torch.randn(4, 8),
            "bias": torch.arange(8, dtype=torch.float32),
            "step": 7,
        }
        writer = S3RdmaStorageWriter("s3://bucket/ckpt", client=client)
        dcp.save(state, storage_writer=writer)

        self.assertTrue(client.exists("ckpt/.metadata"))

        loaded = {
            "weight": torch.zeros(4, 8),
            "bias": torch.zeros(8, dtype=torch.float32),
            "step": 0,
        }
        reader = S3RdmaStorageReader("s3://bucket/ckpt", client=client)
        dcp.load(loaded, storage_reader=reader)

        self.assertEqual(state["weight"], loaded["weight"])
        self.assertEqual(state["bias"], loaded["bias"])
        self.assertEqual(state["step"], loaded["step"])


if __name__ == "__main__":
    run_tests()

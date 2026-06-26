# mypy: ignore-errors
import dataclasses
import io
import os
import pickle
import threading
from dataclasses import dataclass
from urllib.parse import urlparse

import torch
from torch import Tensor
from torch.cuda import cuobj
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint.filesystem import (
    _generate_uuid,
    _StoragePrefix,
    CURRENT_DCP_VERSION,
    DEFAULT_SUFFIX,
)
from torch.distributed.checkpoint.metadata import Metadata, StorageMeta
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    SavePlan,
    SavePlanner,
    WriteItemType,
    WriteResult,
)
from torch.distributed.checkpoint.storage import StorageReader, StorageWriter
from torch.futures import Future


__all__ = [
    "S3RdmaStorageWriter",
    "S3RdmaStorageReader",
    "BotoCuObjClient",
]

_METADATA_KEY = ".metadata"


@dataclass
class _RdmaObjectInfo:
    """Per write-item entry stored in ``Metadata.storage_data``.

    Each saved tensor/shard is a single S3 object holding the raw, contiguous
    bytes of that chunk. ``offsets``/``sizes`` describe the chunk so that a
    later (possibly resharded) load can narrow to a sub-region.
    """

    relative_path: str
    length: int
    offsets: tuple[int, ...] | None = None
    sizes: tuple[int, ...] | None = None


class ObjectClient:
    """Minimal object-store interface the RDMA storage layers depend on.

    The data plane (:meth:`put_rdma`/:meth:`get_rdma`) transfers a registered,
    contiguous buffer directly over RDMA; the control plane carries the
    ``x-amz-rdma-token`` header. ``put_bytes``/``get_bytes`` are the ordinary
    (non-RDMA) path used for small blobs such as the checkpoint metadata.
    """

    def put_rdma(self, key: str, tensor: Tensor) -> None:
        raise NotImplementedError

    def get_rdma(self, key: str, tensor: Tensor) -> None:
        raise NotImplementedError

    def put_bytes(self, key: str, data: bytes) -> None:
        raise NotImplementedError

    def get_bytes(self, key: str) -> bytes:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError


class BotoCuObjClient(ObjectClient):
    """:class:`ObjectClient` backed by boto3 and NVIDIA cuObject.

    For each transfer the buffer is registered with cuObject, an RDMA descriptor
    is minted, and the descriptor is injected as the signed ``x-amz-rdma-token``
    header (``<descriptor>:<hex buffer addr>:<hex size>``) via a botocore
    ``before-sign`` hook -- SigV4 signs all ``x-amz-*`` headers, so it must be
    added before signing. The S3 endpoint transfers the payload directly into or
    out of the registered buffer over RDMA; the HTTP body is empty.

    The boto3 client is configured to match the S3-over-RDMA wire contract:
    unsigned payload (the body is empty; data moves over RDMA) and no default
    content checksum (botocore would otherwise checksum the empty body and the
    server's checksum of the RDMA-delivered bytes would not match).
    """

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        region: str | None = "us-east-1",
        boto3_session=None,
        client_kwargs: dict | None = None,
    ) -> None:
        import boto3
        from botocore.client import Config

        self.bucket = bucket
        session = boto3_session or boto3.session.Session()
        self.s3 = session.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            config=Config(
                signature_version="s3v4",
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
                s3={"addressing_style": "path", "payload_signing_enabled": False},
            ),
            **(client_kwargs or {}),
        )
        self._tls = threading.local()
        self.s3.meta.events.register("before-sign.s3.PutObject", self._inject_token)
        self.s3.meta.events.register("before-sign.s3.GetObject", self._inject_token)

    def _inject_token(self, request, **kwargs) -> None:
        token = getattr(self._tls, "rdma_token", None)
        if token is not None:
            request.headers["x-amz-rdma-token"] = token

    @staticmethod
    def _check_rdma_reply(response) -> None:
        headers = response["ResponseMetadata"]["HTTPHeaders"]
        reply = headers.get("x-amz-rdma-reply")
        if not reply or reply == "501":
            raise RuntimeError(
                "S3 endpoint declined RDMA (x-amz-rdma-reply="
                f"{reply!r}); RDMA is not available for this object store"
            )

    def _rdma(self, key: str, tensor: Tensor, is_put: bool) -> None:
        storage = tensor.untyped_storage()
        nbytes = tensor.nbytes
        addr = storage.data_ptr()
        cuobj.register_buffer(storage)
        try:
            descriptor = cuobj.get_rdma_token(storage, nbytes, 0, is_put)
            try:
                self._tls.rdma_token = f"{descriptor}:{addr:016x}:{nbytes:016x}"
                if is_put:
                    resp = self.s3.put_object(Bucket=self.bucket, Key=key, Body=b"")
                else:
                    resp = self.s3.get_object(Bucket=self.bucket, Key=key)
                    resp["Body"].read()
                self._check_rdma_reply(resp)
            finally:
                self._tls.rdma_token = None
                cuobj.put_rdma_token(descriptor)
        finally:
            cuobj.deregister_buffer(storage)

    def put_rdma(self, key: str, tensor: Tensor) -> None:
        self._rdma(key, tensor, is_put=True)

    def get_rdma(self, key: str, tensor: Tensor) -> None:
        self._rdma(key, tensor, is_put=False)

    def put_bytes(self, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data)

    def get_bytes(self, key: str) -> bytes:
        return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False


def _parse_s3_url(checkpoint_id: str | os.PathLike) -> tuple[str, str]:
    parsed = urlparse(str(checkpoint_id))
    if parsed.scheme != "s3":
        raise ValueError(f"Expected an s3:// URL, got {checkpoint_id!r}")
    return parsed.netloc, parsed.path.strip("/")


def _stage_for_put(tensor: Tensor) -> Tensor:
    """Return a contiguous, RDMA-able host copy of ``tensor``.

    v1 stages through host memory so it works without GPUDirect RDMA. A future
    revision can register device memory directly to remove the device-to-host
    copy.
    """
    t = tensor.detach()
    if not t.is_cpu:
        t = t.cpu()
    return t.contiguous()


class _S3RdmaBase:
    def __init__(
        self,
        path: str | os.PathLike,
        client: ObjectClient | None,
        endpoint_url: str | None,
        region: str | None,
        client_kwargs: dict | None,
    ) -> None:
        self.bucket, self.prefix = _parse_s3_url(path)
        self.path = str(path)
        self.client = client or BotoCuObjClient(
            self.bucket,
            endpoint_url=endpoint_url,
            region=region,
            client_kwargs=client_kwargs,
        )

    def _key(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name

    @property
    def checkpoint_id(self) -> str | os.PathLike:
        return self.path

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        return str(checkpoint_id).startswith("s3://")


class S3RdmaStorageWriter(_S3RdmaBase, StorageWriter):
    """StorageWriter that writes a DCP checkpoint to S3 over RDMA via cuObject.

    Each tensor/shard is written as a single S3 object holding its raw
    contiguous bytes, transferred directly from a registered buffer over RDMA.
    A ``.metadata`` object written last by the coordinator commits the
    checkpoint (S3 ``PutObject`` is atomic).
    """

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        client: ObjectClient | None = None,
        endpoint_url: str | None = None,
        region: str | None = None,
        client_kwargs: dict | None = None,
        overwrite: bool = True,
    ) -> None:
        super().__init__(path, client, endpoint_url, region, client_kwargs)
        self.overwrite = overwrite
        self.save_id = _generate_uuid()
        self.rank: int | None = None
        self.use_collectives = True

    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        if checkpoint_id:
            self.bucket, self.prefix = _parse_s3_url(checkpoint_id)
            self.path = str(checkpoint_id)
        self.save_id = _generate_uuid()

    def set_up_storage_writer(self, is_coordinator: bool, *args, **kwargs) -> None:
        self.rank = kwargs.get("rank")
        self.use_collectives = kwargs.get("use_collectives", True)

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        if self.rank is not None and not self.use_collectives:
            return dataclasses.replace(
                plan, storage_data=_StoragePrefix(f"__{self.rank}_")
            )
        return plan

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        return [
            dataclasses.replace(plan, storage_data=_StoragePrefix(f"__{i}_"))
            if plan.storage_data is None
            else plan
            for i, plan in enumerate(plans)
        ]

    def write_data(
        self, plan: SavePlan, planner: SavePlanner
    ) -> Future[list[WriteResult]]:
        storage_plan: _StoragePrefix = plan.storage_data
        file_count = 0
        results: list[WriteResult] = []

        for item in plan.items:
            name = f"{storage_plan.prefix}{file_count}{DEFAULT_SUFFIX}"
            file_count += 1
            key = self._key(name)
            data = planner.resolve_data(item)

            if item.type == WriteItemType.BYTE_IO:
                blob = data.getvalue()
                self.client.put_bytes(key, blob)
                results.append(
                    WriteResult(
                        index=item.index,
                        size_in_bytes=len(blob),
                        storage_data=_RdmaObjectInfo(name, len(blob)),
                    )
                )
            else:
                staged = _stage_for_put(data)
                nbytes = staged.nbytes
                self.client.put_rdma(key, staged)
                chunk = item.tensor_data.chunk
                results.append(
                    WriteResult(
                        index=item.index,
                        size_in_bytes=nbytes,
                        storage_data=_RdmaObjectInfo(
                            name,
                            nbytes,
                            tuple(chunk.offsets),
                            tuple(chunk.sizes),
                        ),
                    )
                )

        fut: Future[list[WriteResult]] = Future()
        fut.set_result(results)
        return fut

    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        metadata.version = CURRENT_DCP_VERSION
        storage_md = {}
        for wr_list in results:
            storage_md.update({wr.index: wr.storage_data for wr in wr_list})
        metadata.storage_data = storage_md
        metadata.storage_meta = StorageMeta(
            checkpoint_id=self.checkpoint_id, save_id=self.save_id
        )
        self.client.put_bytes(self._key(_METADATA_KEY), pickle.dumps(metadata))

    def storage_meta(self) -> StorageMeta:
        return StorageMeta(checkpoint_id=self.checkpoint_id, save_id=self.save_id)


class S3RdmaStorageReader(_S3RdmaBase, StorageReader):
    """StorageReader for checkpoints written by :class:`S3RdmaStorageWriter`."""

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        client: ObjectClient | None = None,
        endpoint_url: str | None = None,
        region: str | None = None,
        client_kwargs: dict | None = None,
    ) -> None:
        super().__init__(path, client, endpoint_url, region, client_kwargs)
        self.storage_data: dict = {}
        self.load_id = _generate_uuid()

    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        self.storage_data = {}
        if checkpoint_id:
            self.bucket, self.prefix = _parse_s3_url(checkpoint_id)
            self.path = str(checkpoint_id)
        self.load_id = _generate_uuid()

    def read_metadata(self, *args, **kwargs) -> Metadata:
        metadata = pickle.loads(self.client.get_bytes(self._key(_METADATA_KEY)))
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        return metadata

    def set_up_storage_reader(
        self, metadata: Metadata, is_coordinator: bool, *args, **kwargs
    ) -> None:
        self.storage_data = metadata.storage_data
        if self.storage_data is None:
            raise AssertionError("storage_data must not be None in metadata")

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        return plan

    def prepare_global_plan(self, plans: list[LoadPlan]) -> list[LoadPlan]:
        return plans

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        for req in plan.items:
            info: _RdmaObjectInfo = self.storage_data[req.storage_index]
            key = self._key(info.relative_path)

            if req.type == LoadItemType.BYTE_IO:
                bio = io.BytesIO(self.client.get_bytes(key))
                bio.seek(0)
                planner.load_bytes(req, bio)
                continue

            target = planner.resolve_tensor(req).detach()
            temp = torch.empty(info.sizes, dtype=target.dtype)
            self.client.get_rdma(key, temp)

            if info.offsets is not None:
                rel_offsets = [
                    r - s for r, s in zip(req.storage_offsets, info.offsets)
                ]
            else:
                rel_offsets = list(req.storage_offsets)
            temp = narrow_tensor_by_index(temp, rel_offsets, req.lengths)

            if target.size() != temp.size():
                raise AssertionError(
                    f"req {req.storage_index} mismatch sizes "
                    f"{target.size()} vs {temp.size()}"
                )
            target.copy_(temp)
            planner.commit_tensor(req, target)

        fut: Future[None] = Future()
        fut.set_result(None)
        return fut

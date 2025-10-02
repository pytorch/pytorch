# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    TensorWriteData,
    WriteItem,
    WriteItemType,
)


aten = torch.ops.aten


class LocalShardsWrapper(torch.Tensor):
    """
    A wrapper class to hold local shards of a DTensor.
    This class is used largely for checkpointing purposes and implicitly subtypes
    the _Checkpointable protocol.
    """

    __slots__ = ["_local_shards", "_storage_meta"]
    _local_shards: list[torch.Tensor]
    _storage_meta: TensorStorageMetadata

    @staticmethod
    def __new__(
        cls, local_shards: list[torch.Tensor], local_offsets: list[tuple[int, ...]]
    ) -> "LocalShardsWrapper":
        assert all(
            tensor.device == local_shards[0].device for tensor in local_shards[1:]
        )

        # if empty shard, we create a empty tensor
        if len(local_shards) == 0:
            r = torch.Tensor._make_wrapper_subclass(
                cls,
                torch.Size([0, 0]),
            )
            r._local_shards = []
            r._storage_meta = TensorStorageMetadata(
                properties=TensorProperties(),
                size=torch.Size([0, 0]),
                chunks=[
                    ChunkStorageMetadata(
                        offsets=torch.Size([0, 0]), sizes=torch.Size([0, 0])
                    )
                ],
            )
            return r

        # we calculate the total tensor size by "concat" on second tensor dimension
        cat_tensor_shape = list(local_shards[0].size())
        if len(local_shards) > 1 and local_shards[0].ndim == 2:  # column-wise sharding
            for shard in local_shards[1:]:
                cat_tensor_shape[1] += shard.size()[1]

        # in cases of sharding optimizer rowwise, we calculate total tensor size by "concat" on first tensor dimension
        if len(local_shards) > 1 and local_shards[0].ndim == 1:  # row-wise sharding
            for shard in local_shards[1:]:
                cat_tensor_shape[0] += shard.size()[0]

        wrapper_properties = TensorProperties.create_from_tensor(local_shards[0])
        wrapper_shape = torch.Size(cat_tensor_shape)
        chunks_meta = [
            ChunkStorageMetadata(
                offsets=torch.Size(offset),
                sizes=shard.size(),
            )
            for shard, offset in zip(local_shards, local_offsets)
        ]

        r = torch.Tensor._make_wrapper_subclass(
            cls,
            torch.Size(cat_tensor_shape),
        )
        r._local_shards = local_shards
        r._storage_meta = TensorStorageMetadata(
            properties=wrapper_properties,
            size=wrapper_shape,
            chunks=chunks_meta,
        )

        return r

    # necessary for ops dispatching from this subclass to its local shards
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        kwargs = kwargs or {}

        dispatcher = {
            torch.ops._c10d_functional.all_gather_into_tensor.default: cls.handle_all_gather_into_tensor,
            torch.ops._c10d_functional.wait_tensor.default: cls.handle_wait_tensor,
            aten._to_copy.default: cls.handle_to_copy,
            aten.view.default: cls.handle_view,
            aten.equal.default: cls.handle_equal,
            aten.detach.default: cls.handle_detach,
            aten.clone.default: cls.handle_clone,
            aten.new_empty.default: cls.handle_new_empty,
            aten.constant_pad_nd.default: cls.handle_constant_pad_nd,
        }

        if func in dispatcher:
            return dispatcher[func](args, kwargs)
        else:
            raise NotImplementedError(
                f"{func} is not supported for LocalShardsWrapper!"
            )

    @staticmethod
    def handle_all_gather_into_tensor(args, kwargs) -> torch.Tensor:
        local_shards = args[0].local_shards()
        if len(local_shards) == 1:
            result_tensor = local_shards[0]
        # 2D CW sharding: concat columns, 1D RW sharding: concat rows
        result_tensor = torch.cat(local_shards, dim=-1)
        return torch.ops._c10d_functional.all_gather_into_tensor.default(
            result_tensor, *args[1:], **kwargs
        )

    @staticmethod
    def handle_wait_tensor(args, kwargs) -> torch.Tensor:
        return torch.ops._c10d_functional.wait_tensor(args[0])

    @staticmethod
    def handle_to_copy(args, kwargs) -> torch.Tensor:
        res_shards_list = [
            aten._to_copy.default(shard, *args[1:], **kwargs)
            for shard in args[0].local_shards()
        ]
        return LocalShardsWrapper(res_shards_list, args[0].local_offsets())

    @staticmethod
    def handle_view(args, kwargs) -> "LocalShardsWrapper":
        view_shape = args[1]
        res_shards_list = []
        if len(args[0].local_shards()) > 1:
            if args[0].local_shards()[0].ndim == 2:
                assert (
                    args[0].storage_metadata().size[0] == view_shape[0]
                    and args[0].storage_metadata().size[1] == view_shape[1]
                )
                # This accounts for a DTensor quirk, when multiple shards are present on a rank, DTensor on
                # init calls view_as() on the global tensor shape
                # will fail because the view shape is not applicable to individual shards.
                res_shards_list = [
                    aten.view.default(shard, shard.shape, **kwargs)
                    for shard in args[0].local_shards()
                ]
            elif args[0].local_shards()[0].ndim == 1:
                assert args[0].storage_metadata().size[0] == view_shape[0]
                # This case is for optimizer sharding as regardless of sharding type, optimizer state is row wise sharded
                res_shards_list = [
                    aten.view.default(shard, shard.shape, **kwargs)
                    for shard in args[0].local_shards()
                ]
            else:
                raise NotImplementedError("No support for view on tensors ndim > 2")
        else:
            # view is called per shard
            res_shards_list = [
                aten.view.default(shard, args[1], **kwargs)
                for shard in args[0].local_shards()
            ]
        return LocalShardsWrapper(res_shards_list, args[0].local_offsets())

    @staticmethod
    def handle_equal(args, kwargs) -> bool:
        """
        LocalShardsWrapper equal impl also checks for equality of storage metadata
        and the order of shards
        """
        a, b = args[0], args[1]
        if len(a.local_shards()) != len(b.local_shards()):
            return False
        if not all(
            aten.equal.default(x, y) for x, y in zip(a.local_shards(), b.local_shards())
        ):
            return False
        if not a.storage_metadata() == b.storage_metadata():
            return False
        return True

    @staticmethod
    def handle_detach(args, kwargs) -> "LocalShardsWrapper":
        self_ls = args[0]
        deatched_local_shards = [
            aten.detach.default(shard) for shard in self_ls.local_shards()
        ]
        self_ls._local_shards = deatched_local_shards
        self_ls._storage_meta.properties.requires_grad = False
        return self_ls

    @staticmethod
    def handle_clone(args, kwargs) -> "LocalShardsWrapper":
        self_ls = args[0]
        desired_memory_format = kwargs.get("memory_format", None)
        if desired_memory_format and desired_memory_format != torch.preserve_format:
            raise NotImplementedError(
                f"{desired_memory_format} is not supported for LocalShardsWrapper!"
            )
        cloned_local_shards = [
            shard.clone(memory_format=desired_memory_format)
            for shard in self_ls._local_shards
        ]
        return LocalShardsWrapper(cloned_local_shards, self_ls.local_offsets())

    @staticmethod
    def handle_new_empty(args, kwargs) -> "LocalShardsWrapper":
        self_ls = args[0]
        return LocalShardsWrapper(
            [torch.empty_like(shard) for shard in self_ls._local_shards],
            self_ls.local_offsets(),
        )

    @staticmethod
    def handle_constant_pad_nd(args, kwargs) -> "LocalShardsWrapper":
        """
        Apply constant padding to LocalShardsWrapper.

        The padding is based off of the following ideas:
        - The resulting wrapper represents the padded version of the logical tensor.
        - Each shard is padded based on the sharding type + dimension that is padded.
            - For instance, CW shards padded on the left most col will have only padding on the first CW shard.
            - Padding the top row will apply to all CW shards.
        """
        self_lsw = args[0]
        pad_spec = args[1]
        pad_value = args[2] if len(args) > 2 else 0.0

        if len(self_lsw.local_shards()) == 0:
            raise NotImplementedError(
                "Padding empty LocalShardsWrapper is not supported."
            )

        local_shards = self_lsw.local_shards()

        if len(local_shards) == 1:
            padded_shard = torch.nn.functional.pad(
                local_shards[0], pad_spec, mode="constant", value=pad_value
            )
            return LocalShardsWrapper([padded_shard], self_lsw.local_offsets())

        padded_shards = list(local_shards)

        if local_shards[0].ndim == 2:
            # 2D Column-wise sharding: [pad_left, pad_right, pad_top, pad_bottom]
            if len(pad_spec) == 2:
                # Single dimension padding happens on the left most column
                pad_spec = pad_spec + [0, 0]

            if len(pad_spec) != 4:
                raise ValueError(
                    f"Padding spec must be of length 4 for 2D tensors, got {len(pad_spec)}"
                )

            pad_left, pad_right, pad_top, pad_bottom = (
                pad_spec[0],
                pad_spec[1],
                pad_spec[2],
                pad_spec[3],
            )

            # Row paddings are applied to all shards.
            if pad_top > 0:
                padded_shards = [
                    torch.nn.functional.pad(
                        shard, [0, 0, pad_top, 0], mode="constant", value=pad_value
                    )
                    for shard in padded_shards
                ]
            if pad_bottom > 0:
                padded_shards = [
                    torch.nn.functional.pad(
                        shard, [0, 0, 0, pad_bottom], mode="constant", value=pad_value
                    )
                    for shard in padded_shards
                ]

            # Column paddings are only applied to the first/last shard.
            if pad_left > 0:
                padded_shards[0] = torch.nn.functional.pad(
                    padded_shards[0],
                    [pad_left, 0, 0, 0],
                    mode="constant",
                    value=pad_value,
                )
            if pad_right > 0:
                padded_shards[-1] = torch.nn.functional.pad(
                    padded_shards[-1],
                    [0, pad_right, 0, 0],
                    mode="constant",
                    value=pad_value,
                )
        elif local_shards[0].ndim == 1:
            # 1D Row-wise sharding: [pad_top, pad_bottom]
            if len(pad_spec) != 2:
                raise ValueError(
                    f"Padding spec must be of length 2 for 1D tensors, got {len(pad_spec)}"
                )
            pad_top, pad_bottom = pad_spec[0], pad_spec[1]

            if pad_top > 0:
                padded_shards[0] = torch.nn.functional.pad(
                    padded_shards[0], [pad_top, 0], mode="constant", value=pad_value
                )
            if pad_bottom > 0:
                padded_shards[-1] = torch.nn.functional.pad(
                    padded_shards[-1], [0, pad_bottom], mode="constant", value=pad_value
                )
        else:
            raise NotImplementedError(
                f"Padding for {local_shards[0].ndim}D tensors is not supported. "
                f"Only 1D and 2D tensors are currently supported."
            )

        # Update offsets and storage metadata
        original_storage = self_lsw.storage_metadata()
        updated_offsets, updated_storage = LocalShardsWrapper._compute_updated_metadata(
            original_storage,
            self_lsw.local_offsets(),
            pad_spec,
            local_shards[0].ndim,
            padded_shards,
        )

        result = LocalShardsWrapper(padded_shards, updated_offsets)
        result._storage_meta = updated_storage
        return result

    @staticmethod
    def _compute_updated_metadata(
        original_storage: TensorStorageMetadata,
        original_offsets: list[torch.Size],
        pad_spec: list[int],
        ndim: int,
        padded_shards: list[torch.Tensor],
    ) -> tuple[list[tuple[int, ...]], TensorStorageMetadata]:
        """
        Compute updated offsets and storage metadata after padding is applied.

        Args:
            original_storage: Original storage metadata
            original_offsets: Original shard offsets
            pad_spec: Padding specification
            ndim: Number of dimensions (1=RW or 2=CW)
            padded_shards: Padded shard tensors

        Returns:
            Tuple of (updated_offsets, updated_storage_metadata)
        """
        if ndim == 1:  # 1D RW
            pad_top, pad_bottom = pad_spec[0], pad_spec[1]

            updated_offsets = []
            for i, offset in enumerate(original_offsets):
                if i == 0:
                    # First shard: offset stays the same (absorbs top padding)
                    updated_offsets.append(tuple(offset))
                else:
                    # Subsequent shards: shift by top padding amount
                    new_offset = (offset[0] + pad_top,)
                    updated_offsets.append(new_offset)

            new_global_size = torch.Size(
                [original_storage.size[0] + pad_top + pad_bottom]
            )

        elif ndim == 2:  # 2D CW
            pad_left, pad_right, pad_top, pad_bottom = (
                pad_spec[0],
                pad_spec[1],
                pad_spec[2],
                pad_spec[3],
            )

            updated_offsets = []
            for i, offset in enumerate(original_offsets):
                row_offset = offset[0]
                col_offset = offset[1]

                # Top/bottom padding doesn't affect offsets
                # Left padding affects column offsets
                if i == 0:
                    # First shard: column offset stays the same (absorbs left padding)
                    new_2d_offset = (row_offset, col_offset)
                else:
                    # Subsequent shards: shift column offset by left padding amount
                    new_2d_offset = (row_offset, col_offset + pad_left)

                updated_offsets.append(new_2d_offset)

            new_global_size = torch.Size(
                [
                    original_storage.size[0] + pad_top + pad_bottom,
                    original_storage.size[1] + pad_left + pad_right,
                ]
            )

        else:
            raise NotImplementedError(f"Metadata computation for {ndim}D not supported")

        updated_chunks = [
            ChunkStorageMetadata(
                offsets=torch.Size(offset),
                sizes=shard.size(),
            )
            for offset, shard in zip(updated_offsets, padded_shards)
        ]

        updated_storage = TensorStorageMetadata(
            properties=original_storage.properties,
            size=new_global_size,
            chunks=updated_chunks,
        )

        return updated_offsets, updated_storage

    @property
    def device(self) -> torch._C.device:  # type: ignore[override]
        return (
            self._local_shards[0].device if self._local_shards else torch.device("meta")
        )

    @property
    def is_meta(self) -> bool:  # type: ignore[override]
        return self._local_shards[0].is_meta if self._local_shards else True

    def is_pinned(self) -> bool:  # type: ignore[override]
        return self._storage_meta.properties.pin_memory

    def requires_grad_(self, requires_grad: bool = True) -> "LocalShardsWrapper":
        self._storage_meta.properties.requires_grad = requires_grad
        [shard.requires_grad_(requires_grad) for shard in self._local_shards]
        return self

    def local_shards(self) -> list[torch.Tensor]:
        """
        Returns a list of :class:`torch.Tensor' corresponding to the
        local shards for this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return self._local_shards

    def local_sizes(self) -> list[torch.Size]:
        """
        Returns a list of :class:`torch.Size' corresponding to the
        local sizes for the shards on this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return [chunk.sizes for chunk in self._storage_meta.chunks]

    def local_offsets(self) -> list[torch.Size]:
        """
        Returns a list of :class:`torch.Size' corresponding to the
        local offsets for the shards on this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return [chunk.offsets for chunk in self._storage_meta.chunks]

    @property
    def local_chunks(self) -> list[ChunkStorageMetadata]:
        """
        Returns a :class:`list[ChunkStorageMetadata]` object corresponding to the
        metadata for each tensor shard
        """
        return self._storage_meta.chunks

    def storage_metadata(self) -> TensorStorageMetadata:
        """
        Returns a :class:`TensorStorageMetadata` object corresponding to the
        metadata for the local tensor on current rank
        """
        return self._storage_meta

    def is_empty_shard(self) -> bool:
        """
        Returns a :class:`bool` object indicating if the local tensor on current rank
        is an empty tensor
        """
        return self._storage_meta.size[0] == 0 and self._storage_meta.size[1] == 0

    def __create_write_items__(self, fqn: str, object: Any) -> list[WriteItem]:
        """
        For compatibility with DCP, we support creation of WriteItems
        such that they can be saved properly.
        """
        return [
            WriteItem(
                index=MetadataIndex(fqn, chunks.offsets),
                type=WriteItemType.SHARD,
                tensor_data=TensorWriteData(
                    chunk=ChunkStorageMetadata(
                        offsets=chunks.offsets,
                        sizes=chunks.sizes,
                    ),
                    properties=self._storage_meta.properties,
                    size=object.size(),
                ),
            )
            for tensor, chunks in zip(self.local_shards(), self.local_chunks)
        ]

    def __create_chunk_list__(self) -> list[ChunkStorageMetadata]:
        """
        For compatibility with DCP, we support creation of chunk lists
        such that they can be saved properly.
        """
        return self._storage_meta.chunks

    def __get_tensor_shard__(self, index: MetadataIndex) -> torch.Tensor:
        """
        For compatibility with DCP, we support finding shard based on index
        Return a 'torch.Tensor' shard based on 'MetadataIndex'.
        """
        # Fast lookup path
        if index.index is not None:
            if (
                len(self._local_shards) > index.index
                and self._storage_meta.chunks[index.index].offsets == index.offset
            ):
                return self._local_shards[index.index]

        if index.offset is not None:
            for shard, chunk in zip(self._local_shards, self._storage_meta.chunks):
                if chunk.offsets == index.offset:
                    return shard

        # Empty shard case
        if len(self._local_shards) == 0 and self._storage_meta.chunks[
            0
        ].sizes == torch.Size([0, 0]):
            return torch.empty(0)

        raise ValueError(
            f"Could not find shard at '{index.offset}' for FQN: '{index.fqn}'"
        )

    def _get_tensor_size_bytes(self) -> int:
        object_size = 0
        for shard in self.local_shards():
            object_size += shard.nelement() * shard.element_size()
        return object_size

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:  # type: ignore[override]
        return f"LocalShardsWrapper:{self._local_shards} {self._storage_meta}"

    def __str__(self) -> str:
        return f"LocalShardsWrapper:{self._local_shards} {self._storage_meta}"

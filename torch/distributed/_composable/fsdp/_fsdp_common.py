from dataclasses import dataclass
from typing import cast, Optional

import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh


@dataclass
class DataParallelMeshInfo:
    mesh: DeviceMesh
    shard_mesh_dim: Optional[int] = None
    replicate_mesh_dim: Optional[int] = None

    def __post_init__(self):
        if self.shard_mesh_dim is None and self.replicate_mesh_dim is None:
            raise AssertionError(
                "At least one of shard_mesh_dim and replicate_mesh_dim must not be None"
            )


@dataclass
class FSDPMeshInfo(DataParallelMeshInfo):
    def __post_init__(self):
        super().__post_init__()
        if self.shard_mesh_dim is None:
            raise AssertionError("Expects non-None shard_mesh_dim")
        self.shard_mesh_size: int = self.mesh.size(self.shard_mesh_dim)
        self.shard_process_group = cast(
            dist.ProcessGroup, self.mesh.get_group(self.shard_mesh_dim)
        )
        self.shard_mesh_rank: int = self.shard_process_group.rank()


@dataclass
class DDPMeshInfo(DataParallelMeshInfo):
    def __post_init__(self):
        super().__post_init__()
        if self.replicate_mesh_dim is None:
            raise AssertionError("Expects non-None replicate_mesh_dim")
        self.replicate_mesh_size: int = self.mesh.size(self.replicate_mesh_dim)
        self.replicate_process_group = cast(
            dist.ProcessGroup, self.mesh.get_group(self.replicate_mesh_dim)
        )
        self.replicate_mesh_rank: int = self.replicate_process_group.rank()


@dataclass
class HSDPMeshInfo(FSDPMeshInfo, DDPMeshInfo):
    def __post_init__(self):
        super(FSDPMeshInfo, self).__post_init__()
        super(DDPMeshInfo, self).__post_init__()

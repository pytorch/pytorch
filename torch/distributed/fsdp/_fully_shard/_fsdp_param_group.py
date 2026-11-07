"""Private name for a ``fully_shard`` parameter group, which is an engine bucket
(see ``_fsdp_engine``). torchtitan's FSDP lifecycle helpers import it until
FSDP has public APIs for what they do (pytorch/pytorch#203176,
pytorch/pytorch#203177)."""

from ._fsdp_engine import FullyShardBucket as FSDPParamGroup


__all__ = ["FSDPParamGroup"]

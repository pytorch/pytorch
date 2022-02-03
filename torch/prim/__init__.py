
# See the [primTorch Build] note for how these imports work
# Note: they must be ignored by linters
from torch.prim.aliases import alias_infos  # type: ignore

__all__ = ['alias_infos']

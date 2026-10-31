# mypy: allow-untyped-defs

# The fake backend lives in torch.distributed._backends.fake and is registered
# through the "torch.distributed.backends" entry point. Importing this module
# still registers it, as it did before the move.
from torch.distributed._backends.fake import (
    _create_fake_pg,  # noqa: F401
    _register_fake_backend,
    FakeProcessGroup,
    FakeStore,
)


__all__ = ["FakeProcessGroup", "FakeStore"]

_register_fake_backend()

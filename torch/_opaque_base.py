import logging


log = logging.getLogger(__name__)

log.warning("torch._opaque_base is deprecated, use torch._custom_class_base instead")

from torch._custom_class_base import (  # noqa: F401
    _install_opaque_base,
    _rebuild_opaque_base,
    CustomClassBase as OpaqueBase,
    CustomClassBaseMeta as OpaqueBaseMeta,
)

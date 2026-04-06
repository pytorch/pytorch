from __future__ import annotations

from . import ir_base as _ir_base
from . import ir_compute as _ir_compute
from . import ir_views as _ir_views
from . import ir_containers as _ir_containers
from . import ir_extern as _ir_extern

# Populate split module globals that are only needed at runtime and would
# otherwise create import cycles at module import time.
_ir_base.Pointwise = _ir_compute.Pointwise
_ir_base.Reduction = _ir_compute.Reduction
_ir_base.Scan = _ir_compute.Scan
_ir_base.Sort = _ir_compute.Sort
_ir_base._fixed_indexer = _ir_compute._fixed_indexer
_ir_base.nop_loader_fn = _ir_compute.nop_loader_fn
_ir_base.BaseView = _ir_views.BaseView
_ir_base.ExpandView = _ir_views.ExpandView
_ir_base.ReinterpretView = _ir_views.ReinterpretView
_ir_base.as_storage_and_layout = _ir_views.as_storage_and_layout
_ir_base.is_storage_and_layout = _ir_views.is_storage_and_layout
_ir_base.MutableBox = _ir_containers.MutableBox
_ir_base.StorageBox = _ir_containers.StorageBox
_ir_base.TensorBox = _ir_containers.TensorBox
_ir_base.AssertScalar = _ir_extern.AssertScalar
_ir_base.DynamicScalar = _ir_extern.DynamicScalar
_ir_base.EffectfulKernel = _ir_extern.EffectfulKernel
_ir_base.ExternKernel = _ir_extern.ExternKernel
_ir_base.InputsKernel = _ir_extern.InputsKernel
_ir_base.OpaqueMultiOutput = _ir_extern.OpaqueMultiOutput
_ir_base.TemplateBuffer = _ir_extern.TemplateBuffer
_ir_compute.View = _ir_views.View
_ir_compute.as_storage_and_layout = _ir_views.as_storage_and_layout
_ir_compute.TensorBox = _ir_containers.TensorBox
_ir_views.StorageBox = _ir_containers.StorageBox
_ir_views.TensorBox = _ir_containers.TensorBox
_ir_views.ExternKernel = _ir_extern.ExternKernel

from .ir_base import *  # noqa: F401,F403
from .ir_compute import *  # noqa: F401,F403
from .ir_views import *  # noqa: F401,F403
from .ir_containers import *  # noqa: F401,F403
from .ir_extern import *  # noqa: F401,F403

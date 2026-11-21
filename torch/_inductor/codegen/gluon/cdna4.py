# mypy: allow-untyped-defs
"""CDNA4 (AMD gfx950) target bindings for Gluon templates.

The intrinsics a CDNA4 kernel body uses: the MFMA layout and matmul, and the
``buffer_load_to_shared`` async-copy namespace. See ``GLUON_BASE_IMPORTS`` in
gluon_template.py for what belongs here versus in the shared layer.
"""

from .gluon_template import GluonTemplate, GluonTemplateKernel


CDNA4_IMPORTS = """
from triton.experimental.gluon.language.amd import AMDMFMALayout
from triton.experimental.gluon.language.amd.cdna4 import mfma as mfma_cdna4
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async
"""


class Cdna4GluonTemplateKernel(GluonTemplateKernel):
    target_imports = CDNA4_IMPORTS


class Cdna4GluonTemplate(GluonTemplate):
    kernel_type = Cdna4GluonTemplateKernel

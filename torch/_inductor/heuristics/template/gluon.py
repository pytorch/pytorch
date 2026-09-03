# Installs the choices handler that reads config.gluon_flex_attention, on import,
# the way TLX does -- so that config is the single switch and a caller does not
# also have to know which choices class implements the seam.
#
# The factory has to be lazy: this module is imported during choices.py's own
# import (via heuristics/template/__init__), so InductorChoices does not exist yet
# and GluonInductorChoices cannot be constructed until later.
#
# ROCm-only because installing a factory is what gives the handler a uuid() in the
# cache key, and a build with no Gluon body should not pay a key change for it.
# Gluon itself need not be present in the installed Triton: the frontend emits its
# gluon imports into generated kernels as text, so a ROCm build on trunk Triton
# loads this and simply offers no candidate.
import logging

import torch

from torch._inductor import config


log = logging.getLogger(__name__)


def _gluon_inductor_choices():
    from torch._inductor.kernel.flex.gluon_flex import GluonInductorChoices

    return GluonInductorChoices()


if torch.version.hip:
    if config.inductor_choices_class is None:
        config.inductor_choices_class = _gluon_inductor_choices
    else:
        # Someone else owns the seam; theirs wins. Gluon then offers nothing even
        # with the config on, so say so rather than fail silently.
        log.debug(
            "config.inductor_choices_class is already set to %s, so the Gluon "
            "flex-attention choices were not installed; subclass "
            "GluonInductorChoices to offer both.",
            config.inductor_choices_class,
        )

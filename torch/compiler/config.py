"""
This is the top-level configuration module for the compiler, containing
cross-cutting configuration options that affect all parts of the compiler
stack.

You may also be interested in the per-component configuration modules, which
contain configuration options that affect only a specific part of the compiler:

* :mod:`torch._dynamo.config`
* :mod:`torch._inductor.config`
* :mod:`torch._functorch.config`
* :mod:`torch.fx.experimental.config`
"""

import sys
from typing import Optional

from torch.utils._config_module import Config, install_config_module


__all__ = [
    "job_id",
]


# NB: Docblocks go UNDER variable definitions!  Use spacing to make the
# grouping clear.

# FB-internal note: you do NOT have to specify this explicitly specify this if
# you run on MAST, we will automatically default this to
# mast:MAST_JOB_NAME:MAST_JOB_VERSION.
job_id: Optional[str] = Config(env_name_default="TORCH_COMPILE_JOB_ID", default=None)
"""
Semantically, this should be an identifier that uniquely identifies, e.g., a
training job.  You might have multiple attempts of the same job, e.g., if it was
preempted or needed to be restarted, but each attempt should be running
substantially the same workload with the same distributed topology.  You can
set this by environment variable with :envvar:`TORCH_COMPILE_JOB_ID`.

Operationally, this controls the effect of profile-guided optimization related
persistent state.  PGO state can affect how we perform compilation across
multiple invocations of PyTorch, e.g., the first time you run your program we
may compile twice as we discover what inputs are dynamic, and then PGO will
save this state so subsequent invocations only need to compile once, because
they remember it is dynamic.  This profile information, however, is sensitive
to what workload you are running, so we require you to tell us that two jobs
are *related* (i.e., are the same workload) before we are willing to reuse
this information.  Notably, PGO does nothing (even if explicitly enabled)
unless a valid ``job_id`` is available.  In some situations, PyTorch can
configured to automatically compute a ``job_id`` based on the environment it
is running in.

Profiles are always collected on a per rank basis, so different ranks may have
different profiles.  If you know your workload is truly SPMD, you can run with
:data:`torch._dynamo.config.enable_compiler_collectives` to ensure nodes get
consistent profiles across all ranks.
"""


cache_key_tag: str = Config(env_name_default="TORCH_COMPILE_CACHE_KEY_TAG", default="")
"""
Tag to be included in the cache key generation for all torch compile caching.
A common use case for such a tag is to break caches.
"""


install_config_module(sys.modules[__name__])

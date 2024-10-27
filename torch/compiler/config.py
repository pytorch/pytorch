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

import os
from typing import Optional


# NB: Docblocks go UNDER variable definitions!  Use spacing to make the
# grouping clear.

workflow_id: Optional[str] = os.environ.get("TORCH_COMPILE_WORKFLOW_ID", None)
"""
Semantically, this should be an identifier that uniquely identifies, e.g., a
training job  (e.g., at Meta, this would be both the MAST Job Name + MAST Job
Version).  You might have multiple runs of the same job, e.g., if it was
preempted or needed to be restarted.

Operationally, this controls the effect of profile-guided optimization related
persistent state on the local filesystem.  PGO state can affect how we perform
compilation across multiple invocations of PyTorch, e.g., the first time you
run your program we may compile twice as we discover what inputs are dynamic,
and then PGO will save this state so subsequent invocations only need to compile
once, because they remember it is dynamic.  This profile information, however,
is sensitive to what workload you are running, so we require you to tell us
that two jobs are *related* (i.e., are the same workload) before we are willing
to reuse this information.  So PGO is not enabled unless a valid job_id is
available.
"""

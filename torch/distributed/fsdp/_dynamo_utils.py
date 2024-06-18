# mypy: allow-untyped-defs
from typing import Set

import torch.nn as nn


def _annotate_modules_for_dynamo(
    module: nn.Module,
    ignored_modules: Set[nn.Module],
    use_orig_params: bool,
):
    """
    Annotates the submodules in ``module`` 's tree, except those in
    ``ignored_modules``, indicating that the submodules are FSDP-managed and
    saving the ``use_orig_params`` setting passed to the FSDP constructor.
    """
    for submodule in module.modules():
        if submodule not in ignored_modules:
            """[note: Dynamo treats FSDP wrapped modules as UnspecializedNNModule]

            Dynamo doesn't get to see this instance (FullyShardedDataParallel) during tracing, since
            it skips tracing all the torch.distributed.fsdp code.
                - Why? Running the FSDP code eagerly avoids lots of issues trying to trace complex hooks, and also
                gets us graph-breaks on FSDP module boundaries which we want anyway for comm ops.
                - However, we _also_ want dynamo to treat the wrapped module inside FSDP 'unspecially' (*),
                and we need a way to indicate to dynamo which modules are wrapped by FSDP.

            (*) UnspecializedNNModules in dynamo are traced-through without any assumptions, and with thorough
            guards.  NNModules otherwise are 'specialized', meaning there is less overhead due to assuming
            their code is well-behaved.

            One particular issue with specialized NNModules for FSDP is that the
            views created for orig_params are captured into the compiled graph on the first iteration, and while
            they are always going to point to the correct flatparameter and give correct results, their order
            of creation influences the order of backward execution, preventing overlap of comm and computation
            during backward.  We need to _use_ the new parameter views created on each forward iteration, in
            order for backward to interleave hooks with compute per layer.  UnspecializedNNModule lets us achieve
            this by capturing the module code more 'functionally' and passing parameters in as inputs each time.
            """
            submodule._is_fsdp_managed_module = True  # type: ignore[assignment]

            # Dynamo only supports FSDP with use_orig_params=True.
            # This is hacky, but I could not think of another way to add an assertion to dynamo
            # for this, since Dynamo skips all the FSDP code frames and thus can't inspect the
            # FSDP module directly
            submodule._fsdp_use_orig_params = use_orig_params  # type: ignore[assignment]

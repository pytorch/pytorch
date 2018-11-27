torch.utils.checkpoint
======================

.. note::
    Checkpointing is implemented by rerunning a forward-pass segment for
    each checkpointed segment during backward.  This can cause persistent
    states like the RNG state to be advanced than they would without
    checkpointing.  By default, checkpointing includes logic to juggle
    the RNG state such that checkpointed passes making use of RNG
    (through dropout for example) have deterministic output as
    compared to non-checkpointed passes.  The logic to stash and restore
    RNG states can incur a moderate performance hit depending on the runtime
    of checkpointed operations.  If deterministic output compared to
    non-checkpointed passes is not required, set the global flag
    ``torch.utils.checkpoint.preserve_rng_state=False`` to omit stashing and
    restoring the RNG state during each checkpoint.

.. currentmodule:: torch.utils.checkpoint
.. autofunction:: checkpoint
.. autofunction:: checkpoint_sequential

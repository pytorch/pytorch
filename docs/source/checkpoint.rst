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
    non-checkpointed passes is not required, supply ``preserve_rng_state=False``
    to ``checkpoint`` or ``checkpoint_sequential`` to omit stashing and
    restoring the RNG state during each checkpoint.

    The stashing logic saves and restores the RNG state for CPU and another
    device type (infer the device type from Tensor arguments excluding CPU
    tensors by ``_infer_device_type``) to the ``run_fn``. If there are multiple
    device, device state will only be saved for devices of a single device type,
    and the remaining devices will be ignored. Consequently, if any checkpointed
    functions involve randomness, this may result in incorrect gradients. (Note
    that if CUDA devices are among the devices detected, it will be prioritized;
    otherwise, the first device encountered will be selected.) If there are no
    CPU-tensors, the default device type state (default value is `cuda`, and it
    could be set to other device by ``DefaultDeviceType``) will be saved and restored.
    However, the logic has no way to anticipate if the user will move
    Tensors to a new device within the ``run_fn`` itself.  Therefore, if you move
    Tensors to a new device ("new" meaning not belonging to the set of
    [current device + devices of Tensor arguments]) within ``run_fn``, deterministic
    output compared to non-checkpointed passes is never guaranteed.

.. currentmodule:: torch.utils.checkpoint
.. autofunction:: checkpoint
.. autofunction:: checkpoint_sequential

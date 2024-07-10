.. _miscellaneous_environment_variables:

Miscellaneous Environment Variables
===================================
.. list-table::
  :header-rows: 1

  * - Variable
    - Description
  * - ``TORCH_FORCE_WEIGHTS_ONLY_LOAD``
    - If set to [``1``, ``y``, ``yes``, ``true``], the torch.load will use ``weight_only=True``. For more documentation on this, see :func:`torch.load`.
  * - ``TORCH_AUTOGRAD_SHUTDOWN_WAIT_LIMIT``
    - Under some conditions, autograd threads can hang on shutdown, therefore we do not wait for them to shutdown indefinitely but rely on timeout that is default set to ``10`` seconds. This environment variable can be used to set the timeout in seconds.
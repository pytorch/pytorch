.. _miscellaneous_environment_variables:

Miscellaneous Environment Variables
===================================
.. list-table::
   :header-rows: 1

   * - Variable
     - Description
   * - ``TORCH_FORCE_WEIGHTS_ONLY_LOAD``
     - If set to [``1``, ``y``, ``yes``, ``true``], the torch.load will use ``weight_only=True``.
   * - ``TORCH_CPP_LOG_LEVEL``
     - Set the log level of c10 logging facility (supports both GLOG and c10 loggers). Valid values are ``INFO``, ``WARNING``, ``ERROR``, and ``FATAL`` or their numerical equivalents ``0``, ``1``, ``2``, and ``3``.
   * - ``TORCH_AUTOGRAD_SHUTDOWN_WAIT_LIMIT``
     - Under some conditions, autograd threads can hang on shutdown, therefore we do not wait for them to shutdown indefinitely but rely on timeout that is default set to ``10`` seconds. This environment variable can be used to set the timeout in seconds.
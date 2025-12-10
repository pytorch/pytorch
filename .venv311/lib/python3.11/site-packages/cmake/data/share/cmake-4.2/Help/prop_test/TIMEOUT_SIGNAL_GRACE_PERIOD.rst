TIMEOUT_SIGNAL_GRACE_PERIOD
---------------------------

.. versionadded:: 3.27

If the :prop_test:`TIMEOUT_SIGNAL_NAME` test property is set, this property
specifies the number of seconds to wait for a test process to terminate after
sending the custom signal.  Otherwise, this property has no meaning.

The grace period may be any real value greater than ``0.0``, but not greater
than ``60.0``.  If this property is not set, the default is ``1.0`` second.

This is available only on platforms supporting POSIX signals.
It is not available on Windows.

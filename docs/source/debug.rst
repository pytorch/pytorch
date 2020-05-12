torch.utils.debug
=================

.. note::
    These functions are provided for debugging only.
    They will set a process-wide flag.

CPP Stacktraces
^^^^^^^^^^^^^^^

These functions allow you to print the complete cpp stacktrace when
an error happens.

.. autofunction:: is_cpp_stacktraces_enabled
.. autofunction:: set_cpp_stacktraces_enabled

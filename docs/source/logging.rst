.. role:: hidden
    :class: hidden-section

torch._logging
==============

PyTorch has a configurable logging system, where different components can be
given different log level settings. For instance, one component's log messages
can be completely disabled, while another component's log messages can be
set to maximum verbosity.

.. warning:: This feature is a prototype and may have compatibility breaking
    changes in the future.

.. warning:: This feature has not been expanded to control the log messages of
    all components in PyTorch yet.

.. automodule:: torch._logging
.. currentmodule:: torch._logging

.. autosummary::
    :toctree: generated
    :nosignatures:

    set_logs

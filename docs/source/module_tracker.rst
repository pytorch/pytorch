torch.utils.module_tracker
===================================
.. automodule:: torch.utils.module_tracker

This utility can be used to track the current position inside an :class:`torch.nn.Module` hierarchy.
It can be used within other tracking tools to be able to easily associate measured quantities to user-friendly names. This is used in particular in the FlopCounterMode today.

.. autoclass:: torch.utils.module_tracker.ModuleTracker

.. currentmodule:: torch.cuda.tunable

TunableOp
=========

.. note::
    This is a prototype feature, which means it is at an early stage
    for feedback and testing, and its components are subject to change.

Overview
--------

.. automodule:: torch.cuda.tunable

API Reference
-------------

.. autofunction:: enable
.. autofunction:: is_enabled
.. autofunction:: tuning_enable
.. autofunction:: tuning_is_enabled
.. autofunction:: record_untuned_enable
.. autofunction:: record_untuned_is_enabled
.. autofunction:: set_max_tuning_duration
.. autofunction:: get_max_tuning_duration
.. autofunction:: set_max_tuning_iterations
.. autofunction:: get_max_tuning_iterations
.. autofunction:: set_filename
.. autofunction:: get_filename
.. autofunction:: get_results
.. autofunction:: get_validators
.. autofunction:: write_file_on_exit
.. autofunction:: write_file
.. autofunction:: read_file
.. autofunction:: tune_gemm_in_file
.. autofunction:: mgpu_tune_gemm_in_file

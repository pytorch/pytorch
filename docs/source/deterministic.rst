torch.utils.deterministic
=========================
.. py:module:: torch.utils.deterministic
.. currentmodule:: torch.utils.deterministic

.. attribute:: fill_uninitialized_memory

    A :class:`bool` that, if True, causes uninitialized memory to be filled with
    a known value when :meth:`torch.use_deterministic_algorithms()` is set to
    ``True``. Floating point and complex values are set to NaN, and integer
    values are set to the maximum value.

    Default: ``True``

    Filling uninitialized memory is detrimental to performance. So if your
    program is valid and does not use uninitialized memory as the input to an
    operation, then this setting can be turned off for better performance and
    still be deterministic.

    The following operations will fill uninitialized memory when this setting is
    turned on:

        * :func:`torch.Tensor.resize_` when called with a tensor that is not
          quantized
        * :func:`torch.empty`
        * :func:`torch.empty_strided`
        * :func:`torch.empty_permuted`
        * :func:`torch.empty_like`
.. currentmodule:: torch

Binary Operations
=================

As you may have seen in the tutorial, :mod:`MaskedTensor` also has binary operations implemented with the caveat
that the masks in two masked tensors must match or else an error will be raised. As noted in the error, if you
need support for a particular operator or have proposed semantics for how they should be behave instead, please open
an issue on Github. For now, we have decided to go with the most conservative implementation to ensure that users
know exactly what is going on and are being intentional about their decisions with masked semantics.

The available binary operators are:

.. autosummary::
    :toctree: generated
    :nosignatures:

    add
    atan2
    arctan2
    bitwise_and
    bitwise_or
    bitwise_xor
    bitwise_left_shift
    bitwise_right_shift
    div
    divide
    floor_divide
    fmod
    logaddexp
    logaddexp2
    mul
    multiply
    nextafter
    remainder
    sub
    subtract
    true_divide
    eq
    ne
    le
    ge
    greater
    greater_equal
    gt
    less_equal
    lt
    less
    maximum
    minimum
    fmax
    fmin
    not_equal

The available inplace binary operators are all of the above **except**:

.. autosummary::
    :toctree: generated
    :nosignatures:

    logaddexp
    logaddexp2
    equal
    fmin
    minimum
    fmax

As always, if you have any feature requests, please file an issue on Github.

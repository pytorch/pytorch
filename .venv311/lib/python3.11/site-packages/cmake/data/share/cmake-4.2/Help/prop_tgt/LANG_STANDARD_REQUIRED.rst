<LANG>_STANDARD_REQUIRED
------------------------

The variations are:

* :prop_tgt:`C_STANDARD_REQUIRED`
* :prop_tgt:`CXX_STANDARD_REQUIRED`
* :prop_tgt:`CUDA_STANDARD_REQUIRED`
* :prop_tgt:`HIP_STANDARD_REQUIRED`
* :prop_tgt:`OBJC_STANDARD_REQUIRED`
* :prop_tgt:`OBJCXX_STANDARD_REQUIRED`

These properties specify whether the value of :prop_tgt:`<LANG>_STANDARD` is a
requirement.  When false or unset, the :prop_tgt:`<LANG>_STANDARD` target
property is treated as optional and may "decay" to a previous standard if the
requested standard is not available.  When ``<LANG>_STANDARD_REQUIRED`` is set
to true, :prop_tgt:`<LANG>_STANDARD` becomes a hard requirement and a fatal
error will be issued if that requirement cannot be met.

Note that the actual language standard used may be higher than that specified
by :prop_tgt:`<LANG>_STANDARD`, regardless of the value of
``<LANG>_STANDARD_REQUIRED``.  In particular,
:ref:`usage requirements <Target Usage Requirements>` or the use of
:manual:`compile features <cmake-compile-features(7)>` can raise the required
language standard above what :prop_tgt:`<LANG>_STANDARD` specifies.

These properties are initialized by the value of the
:variable:`CMAKE_<LANG>_STANDARD_REQUIRED` variable if it is set when a target
is created.

See the :manual:`cmake-compile-features(7)` manual for information on
compile features and a list of supported compilers.

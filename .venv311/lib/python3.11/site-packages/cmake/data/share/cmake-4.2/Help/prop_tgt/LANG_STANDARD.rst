<LANG>_STANDARD
---------------

The variations are:

* :prop_tgt:`C_STANDARD`
* :prop_tgt:`CXX_STANDARD`
* :prop_tgt:`CUDA_STANDARD`
* :prop_tgt:`HIP_STANDARD`
* :prop_tgt:`OBJC_STANDARD`
* :prop_tgt:`OBJCXX_STANDARD`

These properties specify language standard versions which are requested. When a
newer standard is specified than is supported by the compiler, then it will
fallback to the latest supported standard. This "decay" behavior may be
controlled with the :prop_tgt:`<LANG>_STANDARD_REQUIRED` target property.

Note that the actual language standard used may be higher than that specified
by ``<LANG>_STANDARD``, regardless of the value of
:prop_tgt:`<LANG>_STANDARD_REQUIRED`.  In particular,
:ref:`usage requirements <Target Usage Requirements>` or the use of
:manual:`compile features <cmake-compile-features(7)>` can raise the required
language standard above what ``<LANG>_STANDARD`` specifies.

These properties are initialized by the value of the
:variable:`CMAKE_<LANG>_STANDARD` variable if it is set when a target is
created.

For supported values and CMake versions see the respective pages.
To control compiler-specific extensions see :prop_tgt:`<LANG>_EXTENSIONS`.

See the :manual:`cmake-compile-features(7)` manual for information on
compile features and a list of supported compilers.

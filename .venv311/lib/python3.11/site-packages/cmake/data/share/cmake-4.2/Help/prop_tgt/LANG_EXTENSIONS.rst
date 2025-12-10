<LANG>_EXTENSIONS
-----------------

The variations are:

* :prop_tgt:`C_EXTENSIONS`
* :prop_tgt:`CXX_EXTENSIONS`
* :prop_tgt:`CUDA_EXTENSIONS`
* :prop_tgt:`HIP_EXTENSIONS`
* :prop_tgt:`OBJC_EXTENSIONS`
* :prop_tgt:`OBJCXX_EXTENSIONS`

These properties specify whether compiler-specific extensions are requested.

These properties are initialized by the value of the
:variable:`CMAKE_<LANG>_EXTENSIONS` variable if it is set when a target is
created and otherwise by the value of
:variable:`CMAKE_<LANG>_EXTENSIONS_DEFAULT` (see :policy:`CMP0128`).

For supported CMake versions see the respective pages.
To control language standard versions see :prop_tgt:`<LANG>_STANDARD`.

See the :manual:`cmake-compile-features(7)` manual for information on
compile features and a list of supported compilers.

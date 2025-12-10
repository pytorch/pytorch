CXX_STANDARD_REQUIRED
---------------------

.. versionadded:: 3.1

Boolean describing whether the value of :prop_tgt:`CXX_STANDARD` is a requirement.

If this property is set to ``ON``, then the value of the
:prop_tgt:`CXX_STANDARD` target property is treated as a requirement.  If this
property is ``OFF`` or unset, the :prop_tgt:`CXX_STANDARD` target property is
treated as optional and may "decay" to a previous standard if the requested is
not available.  For compilers that have no notion of a standard level, such as
MSVC 1800 (Visual Studio 2013) and lower, this has no effect.

See the :manual:`cmake-compile-features(7)` manual for information on
compile features and a list of supported compilers.

This property is initialized by the value of
the :variable:`CMAKE_CXX_STANDARD_REQUIRED` variable if it is set when a
target is created.

Alternatively, see :ref:`Requiring Language Standards`.

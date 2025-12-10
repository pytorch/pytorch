UNITY_BUILD_CODE_AFTER_INCLUDE
------------------------------

.. versionadded:: 3.16

Code snippet which is included verbatim by the :prop_tgt:`UNITY_BUILD`
feature just after every ``#include`` statement in the generated unity
source files.  For example:

.. code-block:: cmake

  set(after [[
  #if defined(NOMINMAX)
  #undef NOMINMAX
  #endif
  ]])
  set_target_properties(myTarget PROPERTIES
    UNITY_BUILD_CODE_AFTER_INCLUDE "${after}"
  )

See also :prop_tgt:`UNITY_BUILD_CODE_BEFORE_INCLUDE`.

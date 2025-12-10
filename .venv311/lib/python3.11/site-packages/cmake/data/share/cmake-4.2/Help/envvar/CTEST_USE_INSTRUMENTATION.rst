CTEST_USE_INSTRUMENTATION
-------------------------

.. versionadded:: 4.0

.. include:: include/ENV_VAR.rst

.. note::

   This feature is only available when experimental support for instrumentation
   has been enabled by the ``CMAKE_EXPERIMENTAL_INSTRUMENTATION`` gate.

Setting this environment variable to ``1``, ``True``, or ``ON`` enables
:manual:`instrumentation <cmake-instrumentation(7)>` for CTest in
:ref:`Dashboard Client` mode.

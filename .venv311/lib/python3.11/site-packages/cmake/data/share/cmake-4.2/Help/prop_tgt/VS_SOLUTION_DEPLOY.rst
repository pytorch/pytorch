VS_SOLUTION_DEPLOY
------------------

.. versionadded:: 3.18

Specify that the target should be marked for deployment when not targeting
Windows CE, Windows Phone or a Windows Store application.

If the target platform doesn't support deployment, this property won't have
any effect.

:manual:`Generator expressions <cmake-generator-expressions(7)>` are supported.

Examples
^^^^^^^^

Always deploy target ``foo``:

.. code-block:: cmake

  add_executable(foo SHARED foo.cpp)
  set_property(TARGET foo PROPERTY VS_SOLUTION_DEPLOY ON)

Deploy target ``foo`` for all configurations except ``Release``:

.. code-block:: cmake

  add_executable(foo SHARED foo.cpp)
  set_property(TARGET foo PROPERTY VS_SOLUTION_DEPLOY "$<NOT:$<CONFIG:Release>>")

CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT
-------------------------------------------

.. versionadded:: 3.7.1

CMake sets this variable to a ``TRUE`` value when the
:variable:`CMAKE_INSTALL_PREFIX` has just been initialized to
its default value, typically on the first
run of CMake within a new build tree and the :envvar:`CMAKE_INSTALL_PREFIX`
environment variable is not set on the first run of CMake. This can be used
by project code to change the default without overriding a user-provided value:

.. code-block:: cmake

  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set_property(CACHE CMAKE_INSTALL_PREFIX PROPERTY VALUE "/my/default")
  endif()

cmake_minimum_required
----------------------

Require a minimum version of cmake.

.. code-block:: cmake

  cmake_minimum_required(VERSION <min>[...<policy_max>] [FATAL_ERROR])

.. versionadded:: 3.12
  The optional ``<policy_max>`` version behavior; ignored in older CMake.

Sets the minimum required version of cmake for a project.
Also updates the policy settings as explained below.

``<min>`` and the optional ``<policy_max>`` are each CMake versions of the
form ``major.minor[.patch[.tweak]]``, and the ``...`` is literal.

If the running version of CMake is lower than the ``<min>`` required
version it will stop processing the project and report an error.
The optional ``<policy_max>`` version, if specified, must be at least the
``<min>`` version and sets the `Policy Version`_.
If the running version of CMake is older than 3.12, the extra ``...``
dots will be seen as version component separators, resulting in the
``...<max>`` part being ignored and preserving the pre-3.12 behavior
of basing policies on ``<min>``.

This command will set the value of the
:variable:`CMAKE_MINIMUM_REQUIRED_VERSION` variable to ``<min>``.

The ``FATAL_ERROR`` option is accepted but ignored by CMake 2.6 and
higher.  It should be specified so CMake versions 2.4 and lower fail
with an error instead of just a warning.

.. note::
  Call the ``cmake_minimum_required()`` command at the beginning of
  the top-level ``CMakeLists.txt`` file even before calling the
  :command:`project` command.  It is important to establish version
  and policy settings before invoking other commands whose behavior
  they may affect.  See also policy :policy:`CMP0000`.

  Calling ``cmake_minimum_required()`` inside a :command:`function`
  limits some effects to the function scope when invoked.  For example,
  the :variable:`CMAKE_MINIMUM_REQUIRED_VERSION` variable won't be set
  in the calling scope.  Functions do not introduce their own policy
  scope though, so policy settings of the caller *will* be affected
  (see below).  Due to this mix of things that do and do not affect the
  calling scope, calling ``cmake_minimum_required()`` inside a function
  is generally discouraged.

.. _`Policy Version`:

Policy Version
^^^^^^^^^^^^^^

``cmake_minimum_required(VERSION <min>[...<max>])`` implicitly invokes

.. code-block:: cmake

  cmake_policy(VERSION <min>[...<max>])

.. include:: include/POLICY_VERSION.rst

.. include:: include/DEPRECATED_POLICY_VERSIONS.rst

See Also
^^^^^^^^

* :command:`cmake_policy`

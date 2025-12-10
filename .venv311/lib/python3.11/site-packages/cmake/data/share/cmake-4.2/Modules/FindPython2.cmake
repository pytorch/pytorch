# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindPython2
-----------

.. versionadded:: 3.12

Finds Python 2 interpreter, compiler and development environment (include
directories and libraries):

.. code-block:: cmake

  find_package(Python2 [<version>] [COMPONENTS <components>...] [...])

.. versionadded:: 3.19
  When a version is requested, it can be specified as a simple value or as a
  range. For a detailed description of version range usage and capabilities,
  refer to the :command:`find_package` command.

The following components are supported:

* ``Interpreter``: search for Python 2 interpreter
* ``Compiler``: search for Python 2 compiler. Only offered by IronPython.
* ``Development``: search for development artifacts (include directories and
  libraries).

  .. versionadded:: 3.18
    This component includes two sub-components which can be specified
    independently:

    * ``Development.Module``: search for artifacts for Python 2 module
      developments.
    * ``Development.Embed``: search for artifacts for Python 2 embedding
      developments.

.. versionadded:: 3.14

  * ``NumPy``: search for NumPy include directories. Specifying this component
    imply also the components ``Interpreter`` and ``Development.Module``.

  .. versionchanged:: 4.2
    The component ``Development.Module`` is no longer implied when the policy
    :policy:`CMP0201` is set to ``NEW``.

If no ``COMPONENTS`` are specified, ``Interpreter`` is assumed.

If component ``Development`` is specified, it implies sub-components
``Development.Module`` and ``Development.Embed``.

.. versionchanged:: 4.1
  In a cross-compiling mode (i.e. the :variable:`CMAKE_CROSSCOMPILING` variable
  is defined to true), the following constraints, when the policy
  :policy:`CMP0190` is set to ``NEW``, now apply to the requested components:

  * ``Interpreter`` or ``Compiler`` alone: the host artifacts will be searched.
  * ``Interpreter`` or ``Compiler`` with ``Development`` or any sub-component:
    The target artifacts will be searched. In this case, the
    :variable:`CMAKE_CROSSCOMPILING_EMULATOR` variable must be defined and will
    be used to execute the interpreter or the compiler.

  When both host and target artifacts are needed, two different calls to the
  :command:`find_package` command should be done. The
  ``Python_ARTIFACTS_PREFIX`` variable can be helpful in this situation.

To ensure consistent versions between components ``Interpreter``, ``Compiler``,
``Development`` (or one of its sub-components) and ``NumPy``, specify all
components at the same time:

.. code-block:: cmake

  find_package (Python2 COMPONENTS Interpreter Development)

This module looks only for version 2 of Python. This module can be used
concurrently with :module:`FindPython3` module to use both Python versions.

The :module:`FindPython` module can be used if Python version does not matter
for you.

.. note::

  If components ``Interpreter`` and ``Development`` (or one of its
  sub-components) are both specified, this module search only for interpreter
  with same platform architecture as the one defined by CMake
  configuration. This constraint does not apply if only ``Interpreter``
  component is specified.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

.. versionchanged:: 3.14
  :ref:`Imported Targets <Imported Targets>` are only created when
  :prop_gbl:`CMAKE_ROLE` is ``PROJECT``.

``Python2::Interpreter``
  Python 2 interpreter. This target is defined only if the ``Interpreter``
  component is found.
``Python2::InterpreterDebug``
  .. versionadded:: 3.30

  Python 2 debug interpreter. This target is defined only if the
  ``Interpreter`` component is found and the ``Python2_EXECUTABLE_DEBUG``
  variable is defined. The target is only defined on the ``Windows`` platform.

``Python2::InterpreterMultiConfig``
  .. versionadded:: 3.30

  Python 2 interpreter. The release or debug version of the interpreter will be
  used, based on the context (platform, configuration).
  This target is defined only if the ``Interpreter`` component is found

``Python2::Compiler``
  Python 2 compiler. This target is defined only if the ``Compiler`` component
  is found.
``Python2::Module``
  .. versionadded:: 3.15

  Python 2 library for Python module. Target defined if component
  ``Development.Module`` is found.

``Python2::Python``
  Python 2 library for Python embedding. Target defined if component
  ``Development.Embed`` is found.

``Python2::NumPy``
  .. versionadded:: 3.14

  NumPy library for Python 2. Target defined if component ``NumPy`` is found.
  Moreover, this target has the ``Python2::Module`` target as dependency.

  .. versionchanged:: 4.2
    This target does not have anymore the ``Python2::Module`` target as
    dependency when the policy :policy:`CMP0201` is set to ``NEW``.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables
(see :ref:`Standard Variable Names <CMake Developer Standard Variable Names>`):

``Python2_FOUND``
  Boolean indicating whether system has the Python 2 requested components.
``Python2_Interpreter_FOUND``
  Boolean indicating whether system has the Python 2 interpreter.
``Python2_EXECUTABLE``
  Path to the Python 2 interpreter.
``Python2_EXECUTABLE_DEBUG``
  .. versionadded:: 3.30

  Path to the debug Python 2 interpreter. It is only defined on the ``Windows``
  platform.

``Python2_INTERPRETER``
  .. versionadded:: 3.30

  Path to the Python 2 interpreter, defined as a
  :manual:`generator expression <cmake-generator-expressions(7)>` selecting
  the ``Python2_EXECUTABLE`` or ``Python2_EXECUTABLE_DEBUG`` variable based on
  the context (platform, configuration).

``Python2_INTERPRETER_ID``
  A short string unique to the interpreter. Possible values include:
    * Python
    * ActivePython
    * Anaconda
    * Canopy
    * IronPython
    * PyPy
``Python2_STDLIB``
  Standard platform independent installation directory.

  Information returned by ``sysconfig.get_path('stdlib')`` or else
  ``distutils.sysconfig.get_python_lib(plat_specific=False,standard_lib=True)``.
``Python2_STDARCH``
  Standard platform dependent installation directory.

  Information returned by ``sysconfig.get_path('platstdlib')`` or else
  ``distutils.sysconfig.get_python_lib(plat_specific=True,standard_lib=True)``.
``Python2_SITELIB``
  Third-party platform independent installation directory.

  Information returned by ``sysconfig.get_path('purelib')`` or else
  ``distutils.sysconfig.get_python_lib(plat_specific=False,standard_lib=False)``.
``Python2_SITEARCH``
  Third-party platform dependent installation directory.

  Information returned by ``sysconfig.get_path('platlib')`` or else
  ``distutils.sysconfig.get_python_lib(plat_specific=True,standard_lib=False)``.
``Python2_Compiler_FOUND``
  Boolean indicating whether system has the Python 2 compiler.
``Python2_COMPILER``
  Path to the Python 2 compiler. Only offered by IronPython.
``Python2_COMPILER_ID``
  A short string unique to the compiler. Possible values include:
    * IronPython

``Python2_DOTNET_LAUNCHER``
  .. versionadded:: 3.18

  The ``.Net`` interpreter. Only used by ``IronPython`` implementation.

``Python2_Development_FOUND``
  Boolean indicating whether system has the Python 2 development artifacts.

``Python2_Development.Module_FOUND``
  .. versionadded:: 3.18

  Boolean indicating whether system has the Python 2 development artifacts
  for Python module.

``Python2_Development.Embed_FOUND``
  .. versionadded:: 3.18

  Boolean indicating whether system has the Python 2 development artifacts
  for Python embedding.

``Python2_INCLUDE_DIRS``
  The Python 2 include directories.

``Python2_DEBUG_POSTFIX``
  .. versionadded:: 3.30

  Postfix of debug python module. This variable can be used to define the
  :prop_tgt:`DEBUG_POSTFIX` target property.

``Python2_LINK_OPTIONS``
  .. versionadded:: 3.19

  The Python 2 link options. Some configurations require specific link options
  for a correct build and execution.

``Python2_LIBRARIES``
  The Python 2 libraries.
``Python2_LIBRARY_DIRS``
  The Python 2 library directories.
``Python2_RUNTIME_LIBRARY_DIRS``
  The Python 2 runtime library directories.
``Python2_VERSION``
  Python 2 version.
``Python2_VERSION_MAJOR``
  Python 2 major version.
``Python2_VERSION_MINOR``
  Python 2 minor version.
``Python2_VERSION_PATCH``
  Python 2 patch version.

``Python2_PyPy_VERSION``
  .. versionadded:: 3.18

  Python 2 PyPy version.

``Python2_NumPy_FOUND``
  .. versionadded:: 3.14

  Boolean indicating whether system has the NumPy.

``Python2_NumPy_INCLUDE_DIRS``
  .. versionadded:: 3.14

  The NumPy include directories.

``Python2_NumPy_VERSION``
  .. versionadded:: 3.14

  The NumPy version.

Hints
^^^^^

``Python2_ROOT_DIR``
  Define the root directory of a Python 2 installation.

``Python2_USE_STATIC_LIBS``
  * If not defined, search for shared libraries and static libraries in that
    order.
  * If set to TRUE, search **only** for static libraries.
  * If set to FALSE, search **only** for shared libraries.

  .. note::

    This hint will be ignored on ``Windows`` because static libraries are not
    available on this platform.

``Python2_FIND_STRATEGY``
  .. versionadded:: 3.15

  This variable defines how lookup will be done.
  The ``Python2_FIND_STRATEGY`` variable can be set to one of the following:

  * ``VERSION``: Try to find the most recent version in all specified
    locations.
    This is the default if policy :policy:`CMP0094` is undefined or set to
    ``OLD``.
  * ``LOCATION``: Stops lookup as soon as a version satisfying version
    constraints is founded.
    This is the default if policy :policy:`CMP0094` is set to ``NEW``.

  See also ``Python2_FIND_UNVERSIONED_NAMES``.

``Python2_FIND_REGISTRY``
  .. versionadded:: 3.13

  On Windows the ``Python2_FIND_REGISTRY`` variable determine the order
  of preference between registry and environment variables.
  the ``Python2_FIND_REGISTRY`` variable can be set to one of the following:

  * ``FIRST``: Try to use registry before environment variables.
    This is the default.
  * ``LAST``: Try to use registry after environment variables.
  * ``NEVER``: Never try to use registry.

``Python2_FIND_FRAMEWORK``
  .. versionadded:: 3.15

  On macOS the ``Python2_FIND_FRAMEWORK`` variable determine the order of
  preference between Apple-style and unix-style package components.
  This variable can take same values as :variable:`CMAKE_FIND_FRAMEWORK`
  variable.

  .. note::

    Value ``ONLY`` is not supported so ``FIRST`` will be used instead.

  If ``Python2_FIND_FRAMEWORK`` is not defined, :variable:`CMAKE_FIND_FRAMEWORK`
  variable will be used, if any.

``Python2_FIND_VIRTUALENV``
  .. versionadded:: 3.15

  This variable defines the handling of virtual environments managed by
  ``virtualenv`` or ``conda``. It is meaningful only when a virtual environment
  is active (i.e. the ``activate`` script has been evaluated). In this case, it
  takes precedence over ``Python2_FIND_REGISTRY`` and ``CMAKE_FIND_FRAMEWORK``
  variables.  The ``Python2_FIND_VIRTUALENV`` variable can be set to one of the
  following:

  * ``FIRST``: The virtual environment is used before any other standard
    paths to look-up for the interpreter. This is the default.
  * ``ONLY``: Only the virtual environment is used to look-up for the
    interpreter.
  * ``STANDARD``: The virtual environment is not used to look-up for the
    interpreter but environment variable ``PATH`` is always considered.
    In this case, variable ``Python2_FIND_REGISTRY`` (Windows) or
    ``CMAKE_FIND_FRAMEWORK`` (macOS) can be set with value ``LAST`` or
    ``NEVER`` to select preferably the interpreter from the virtual
    environment.

  .. versionadded:: 3.17
    Added support for ``conda`` environments.

  .. note::

    If the component ``Development`` is requested (or one of its
    sub-components) and is not found or the wrong artifacts are returned,
    including also the component ``Interpreter`` may be helpful.

``Python2_FIND_IMPLEMENTATIONS``
  .. versionadded:: 3.18

  This variable defines, in an ordered list, the different implementations
  which will be searched. The ``Python2_FIND_IMPLEMENTATIONS`` variable can
  hold the following values:

  * ``CPython``: this is the standard implementation. Various products, like
    ``Anaconda`` or ``ActivePython``, rely on this implementation.
  * ``IronPython``: This implementation use the ``CSharp`` language for
    ``.NET Framework`` on top of the `Dynamic Language Runtime` (``DLR``).
    See `IronPython <https://ironpython.net>`_.
  * ``PyPy``: This implementation use ``RPython`` language and
    ``RPython translation toolchain`` to produce the python interpreter.
    See `PyPy <https://pypy.org>`_.

  The default value is:

  * Windows platform: ``CPython``, ``IronPython``
  * Other platforms: ``CPython``

  .. note::

    This hint has the lowest priority of all hints, so even if, for example,
    you specify ``IronPython`` first and ``CPython`` in second, a python
    product based on ``CPython`` can be selected because, for example with
    ``Python2_FIND_STRATEGY=LOCATION``, each location will be search first for
    ``IronPython`` and second for ``CPython``.

  .. note::

    When ``IronPython`` is specified, on platforms other than ``Windows``, the
    ``.Net`` interpreter (i.e. ``mono`` command) is expected to be available
    through the ``PATH`` variable.

``Python2_FIND_UNVERSIONED_NAMES``
  .. versionadded:: 3.20

  This variable defines how the generic names will be searched. Currently, it
  only applies to the generic names of the interpreter, namely, ``python2`` and
  ``python``.
  The ``Python2_FIND_UNVERSIONED_NAMES`` variable can be set to one of the
  following values:

  * ``FIRST``: The generic names are searched before the more specialized ones
    (such as ``python2.5`` for example).
  * ``LAST``: The generic names are searched after the more specialized ones.
    This is the default.
  * ``NEVER``: The generic name are not searched at all.

  See also ``Python2_FIND_STRATEGY``.

Artifacts Specification
^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.16

To solve special cases, it is possible to specify directly the artifacts by
setting the following variables:

``Python2_EXECUTABLE``
  The path to the interpreter.

``Python2_COMPILER``
  The path to the compiler.

``Python2_DOTNET_LAUNCHER``
  .. versionadded:: 3.18

  The ``.Net`` interpreter. Only used by ``IronPython`` implementation.

``Python2_LIBRARY``
  The path to the library. It will be used to compute the
  variables ``Python2_LIBRARIES``, ``Python2_LIBRARY_DIRS`` and
  ``Python2_RUNTIME_LIBRARY_DIRS``.

``Python2_INCLUDE_DIR``
  The path to the directory of the ``Python`` headers. It will be used to
  compute the variable ``Python2_INCLUDE_DIRS``.

``Python2_NumPy_INCLUDE_DIR``
  The path to the directory of the ``NumPy`` headers. It will be used to
  compute the variable ``Python2_NumPy_INCLUDE_DIRS``.

.. note::

  All paths must be absolute. Any artifact specified with a relative path
  will be ignored.

.. note::

  When an artifact is specified, all ``HINTS`` will be ignored and no search
  will be performed for this artifact.

  If more than one artifact is specified, it is the user's responsibility to
  ensure the consistency of the various artifacts.

By default, this module supports multiple calls in different directories of a
project with different version/component requirements while providing correct
and consistent results for each call. To support this behavior, CMake cache
is not used in the traditional way which can be problematic for interactive
specification. So, to enable also interactive specification, module behavior
can be controlled with the following variable:

``Python2_ARTIFACTS_INTERACTIVE``
  .. versionadded:: 3.18

  Selects the behavior of the module. This is a boolean variable:

  * If set to ``TRUE``: Create CMake cache entries for the above artifact
    specification variables so that users can edit them interactively.
    This disables support for multiple version/component requirements.
  * If set to ``FALSE`` or undefined: Enable multiple version/component
    requirements.

``Python2_ARTIFACTS_PREFIX``
  .. versionadded:: 4.0

  Define a custom prefix which will be used for the definition of all the
  result variables, targets, and commands. By using this variable, this module
  supports multiple calls in the same directory with different
  version/component requirements.
  For example, in case of cross-compilation, development components are needed
  but the native python interpreter can also be required:

  .. code-block:: cmake

    find_package(Python2 COMPONENTS Development)

    set(Python2_ARTIFACTS_PREFIX "_HOST")
    find_package(Python2 COMPONENTS Interpreter)

    # Here Python2_HOST_EXECUTABLE and Python2_HOST::Interpreter artifacts are defined

  .. note::

    For consistency with standard behavior of modules, the various standard
    ``_FOUND`` variables (i.e. without the custom prefix) are also defined by
    each call to the :command:`find_package` command.

Commands
^^^^^^^^

This module defines the command ``Python2_add_library`` (when
:prop_gbl:`CMAKE_ROLE` is ``PROJECT``), which has the same semantics as
:command:`add_library` and adds a dependency to target ``Python2::Python`` or,
when library type is ``MODULE``, to target ``Python2::Module`` and takes care
of Python module naming rules:

.. code-block:: cmake

  Python2_add_library (<name> [STATIC | SHARED | MODULE]
                       <source1> [<source2> ...])

If library type is not specified, ``MODULE`` is assumed.

.. versionadded:: 3.30
  For ``MODULE`` type, the :prop_tgt:`DEBUG_POSTFIX` target property is
  initialized with the value of ``Python2_DEBUG_POSTFIX`` variable if defined.
#]=======================================================================]


set (_PYTHON_BASE Python2)
if(${_PYTHON_BASE}_ARTIFACTS_PREFIX)
  set(_PYTHON_PREFIX "${_PYTHON_BASE}${${_PYTHON_BASE}_ARTIFACTS_PREFIX}")
else()
  set(_PYTHON_PREFIX "${_PYTHON_BASE}")
endif()

set (_${_PYTHON_PREFIX}_REQUIRED_VERSION_MAJOR 2)

include (${CMAKE_CURRENT_LIST_DIR}/FindPython/Support.cmake)

if (COMMAND __${_PYTHON_PREFIX}_add_library AND NOT COMMAND ${_PYTHON_PREFIX}_add_library)
  cmake_language(EVAL CODE
    "macro (${_PYTHON_PREFIX}_add_library)
       __${_PYTHON_PREFIX}_add_library (${_PYTHON_PREFIX} \${ARGV})
     endmacro()")
endif()

unset (_PYTHON_BASE)
unset (_PYTHON_PREFIX)

cmake_instrumentation
---------------------

.. versionadded:: 4.0

.. note::

   This command is only available when experimental support for instrumentation
   has been enabled by the ``CMAKE_EXPERIMENTAL_INSTRUMENTATION`` gate.

Enables interacting with the
:manual:`CMake Instrumentation API <cmake-instrumentation(7)>`.

This allows for configuring instrumentation at the project-level.

.. code-block:: cmake

  cmake_instrumentation(
    API_VERSION <version>
    DATA_VERSION <version>
    [HOOKS <hooks>...]
    [OPTIONS <options>...]
    [CALLBACK <callback>]
    [CUSTOM_CONTENT <name> <type> <content>]
  )

The ``API_VERSION`` and ``DATA_VERSION`` must always be given.  Currently, the
only supported value for both fields is 1.  See :ref:`cmake-instrumentation API v1`
for details of the ``API_VERSION`` and :ref:`cmake-instrumentation Data v1` for details
of the ``DATA_VERSION``.

Each of the optional keywords ``HOOKS``, ``OPTIONS``, and ``CALLBACK``
correspond to one of the parameters to the :ref:`cmake-instrumentation v1 Query Files`.
The ``CALLBACK`` keyword can be provided multiple times to create multiple callbacks.

Whenever ``cmake_instrumentation`` is invoked, a query file is generated in
``<build>/.cmake/instrumentation/v1/query/generated`` to enable instrumentation
with the provided arguments.

.. _`cmake_instrumentation CUSTOM_CONTENT`:

Custom CMake Content
^^^^^^^^^^^^^^^^^^^^

The ``CUSTOM_CONTENT`` argument specifies certain data from configure time to
include in each :ref:`cmake-instrumentation v1 CMake Content File`. This
may be used to associate instrumentation data with certain information about its
configuration, such as the optimization level or whether it is part of a
coverage build.

``CUSTOM_CONTENT`` expects ``name``, ``type`` and ``content`` arguments.

``name`` is a specifier to identify the content being reported.

``type`` specifies how the content should be interpreted. Supported values are:
  * ``STRING`` the content is a string.
  * ``BOOL`` the content should be interpreted as a boolean. It will be ``true``
    under the same conditions that ``if()`` would be true for the given value.
  * ``LIST`` the content is a CMake ``;`` separated list that should be parsed.
  * ``JSON`` the content should be parsed as a JSON string. This can be a
    number such as ``1`` or ``5.0``, a quoted string such as ``\"string\"``,
    a boolean value ``true``/``false``, or a JSON object such as
    ``{ \"key\" : \"value\" }`` that may be constructed using
    ``string(JSON ...)`` commands.

``content`` is the actual content to report.

Example
^^^^^^^

The following example shows an invocation of the command and its
equivalent JSON query file.

.. code-block:: cmake

  cmake_instrumentation(
    API_VERSION 1
    DATA_VERSION 1
    HOOKS postGenerate preCMakeBuild postCMakeBuild
    OPTIONS staticSystemInformation dynamicSystemInformation trace
    CALLBACK ${CMAKE_COMMAND} -P /path/to/handle_data.cmake
    CALLBACK ${CMAKE_COMMAND} -P /path/to/handle_data_2.cmake
    CUSTOM_CONTENT myString STRING string
    CUSTOM_CONTENT myList   LIST   "item1;item2"
    CUSTOM_CONTENT myObject JSON   "{ \"key\" : \"value\" }"
  )

.. code-block:: json

  {
    "version": 1,
    "hooks": [
      "postGenerate", "preCMakeBuild", "postCMakeBuild"
    ],
    "options": [
      "staticSystemInformation", "dynamicSystemInformation", "trace"
    ],
    "callbacks": [
      "/path/to/cmake -P /path/to/handle_data.cmake"
      "/path/to/cmake -P /path/to/handle_data_2.cmake"
    ]
  }

This will also result in the following content included in each
:ref:`cmake-instrumentation v1 CMake Content File`:

.. code-block:: json

  "custom": {
    "myString": "string",
    "myList": [
      "item1", "item2"
    ],
    "myObject": {
      "key": "value"
    }
  }

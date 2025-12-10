CMAKE_EXPORT_BUILD_DATABASE
---------------------------

.. versionadded:: 3.31

.. note::

   This variable is meaningful only when experimental support for build
   databases has been enabled by the
   ``CMAKE_EXPERIMENTAL_EXPORT_BUILD_DATABASE`` gate.

Enable/Disable output of module compile commands during the build.

If enabled, generates a ``build_database.json`` file containing the
information necessary to compile a target's C++ module sources with any
tooling. The format of the JSON file looks like:

.. code-block:: javascript

  {
    "version": 1,
    "revision": 0,
    "sets": [
      {
        "family-name" : "export_build_database",
        "name" : "export_build_database@Debug",
        "translation-units" : [
          {
            "arguments": [
              "/path/to/compiler",
              "...",
            ],
            "baseline-arguments" :
            [
              "...",
            ],
            "local-arguments" :
            [
              "...",
            ],
            "object": "CMakeFiles/target.dir/source.cxx.o",
            "private": true,
            "provides": {
              "importable": "path/to/bmi"
            },
            "requires" : [],
            "source": "path/to/source.cxx",
            "work-directory": "/path/to/working/directory"
          }
        ],
        "visible-sets" : []
      }
    ]
  }

This is initialized by the :envvar:`CMAKE_EXPORT_BUILD_DATABASE` environment
variable, and initializes the :prop_tgt:`EXPORT_BUILD_DATABASE` target
property for all targets.

.. note::
  This option is implemented only by the :ref:`Ninja Generators`.  It is
  ignored on other generators.

When supported and enabled, numerous targets are created in order to make it
possible to build a file containing just the commands that are needed for the
tool in question.

``cmake_build_database-<CONFIG>``
  Writes ``build_database_<CONFIG>.json``. Writes a build database for the
  entire build for the given configuration and all languages. Not available if
  the configuration name is the empty string.

``cmake_build_database-<LANG>-<CONFIG>``
  Writes ``build_database_<LANG>_<CONFIG>.json``. Writes build database for
  the entire build for the given configuration and language. Not available if
  the configuration name is the empty string.

``cmake_build_database-<LANG>``
  Writes ``build_database_<LANG>.json``. Writes build database for the entire
  build for the given language and all configurations. In a multi-config
  generator, other build configuration database may be assumed to exist.

``cmake_build_database``
  Writes to ``build_database.json``. Writes build database for all languages
  and configurations. In a multi-config generator, other build configuration
  database may be assumed to exist.

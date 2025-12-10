define_property
---------------

Define and document custom properties.

.. code-block:: cmake

  define_property(<GLOBAL | DIRECTORY | TARGET | SOURCE |
                   TEST | VARIABLE | CACHED_VARIABLE>
                   PROPERTY <name> [INHERITED]
                   [BRIEF_DOCS <brief-doc> [docs...]]
                   [FULL_DOCS <full-doc> [docs...]]
                   [INITIALIZE_FROM_VARIABLE <variable>])

Defines one property in a scope for use with the :command:`set_property` and
:command:`get_property` commands. It is mainly useful for defining the way
a property is initialized or inherited. Historically, the command also
associated documentation with a property, but that is no longer considered a
primary use case.

The first argument determines the kind of scope in which the property should
be used.  It must be one of the following:

* ``GLOBAL``          - associated with the global namespace.
* ``DIRECTORY``       - associated with one directory.
* ``TARGET``          - associated with one target.
* ``SOURCE``          - associated with one source file.
* ``TEST``            - associated with a test named with :command:`add_test`.
* ``VARIABLE``        - documents a CMake language variable.
* ``CACHED_VARIABLE`` - documents a CMake cache variable.

Note that unlike :command:`set_property` and :command:`get_property` no
actual scope needs to be given; only the kind of scope is important.

The required ``PROPERTY`` option is immediately followed by the name of
the property being defined.

If the ``INHERITED`` option is given, then the :command:`get_property` command
will chain up to the next higher scope when the requested property is not set
in the scope given to the command.

* ``DIRECTORY`` scope chains to its parent directory's scope, continuing the
  walk up parent directories until a directory has the property set or there
  are no more parents.  If still not found at the top level directory, it
  chains to the ``GLOBAL`` scope.
* ``TARGET``, ``SOURCE`` and ``TEST`` properties chain to ``DIRECTORY`` scope,
  including further chaining up the directories, etc. as needed.

Note that this scope chaining behavior only applies to calls to
:command:`get_property`, :command:`get_directory_property`,
:command:`get_target_property`, :command:`get_source_file_property` and
:command:`get_test_property`.  There is no inheriting behavior when *setting*
properties, so using ``APPEND`` or ``APPEND_STRING`` with the
:command:`set_property` command will not consider inherited values when working
out the contents to append to.

The ``BRIEF_DOCS`` and ``FULL_DOCS`` options are followed by strings to be
associated with the property as its brief and full documentation.
CMake does not use this documentation other than making it available to the
project via corresponding options to the :command:`get_property` command.

.. versionchanged:: 3.23

  The ``BRIEF_DOCS`` and ``FULL_DOCS`` options are optional.

.. versionadded:: 3.23

  The ``INITIALIZE_FROM_VARIABLE`` option specifies a variable from which the
  property should be initialized. It can only be used with target properties.
  The ``<variable>`` name must end with the property name and must not begin
  with ``CMAKE_`` or ``_CMAKE_``. The property name must contain at least one
  underscore. It is recommended that the property name have a prefix specific
  to the project.

Property Redefinition
^^^^^^^^^^^^^^^^^^^^^

Once a property is defined for a particular type of scope, it cannot be
redefined. Attempts to redefine an existing property by calling
:command:`define_property` with the same scope type and property name
will be silently ignored. Defining the same property name for two different
kinds of scope is valid.

:command:`get_property` can be used to determine whether a property is
already defined for a particular kind of scope, and if so, to examine its
definition. For example:

.. code-block:: cmake

  # Initial definition
  define_property(TARGET PROPERTY MY_NEW_PROP
    BRIEF_DOCS "My new custom property"
  )

  # Later examination
  get_property(my_new_prop_exists
    TARGET NONE
    PROPERTY MY_NEW_PROP
    DEFINED
  )

  if(my_new_prop_exists)
    get_property(my_new_prop_docs
      TARGET NONE
      PROPERTY MY_NEW_PROP
      BRIEF_DOCS
    )
    # ${my_new_prop_docs} is now set to "My new custom property"
  endif()

See Also
^^^^^^^^

* :command:`get_property`
* :command:`set_property`

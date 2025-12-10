IMPORTED_GLOBAL
---------------

.. versionadded:: 3.11

Indication of whether an :ref:`IMPORTED target <Imported Targets>` is
globally visible.

The boolean value of this property is True for targets created with the
``IMPORTED`` ``GLOBAL`` options to :command:`add_executable()` or
:command:`add_library()`. It is always False for targets built within the
project.

For targets created with the ``IMPORTED`` option to
:command:`add_executable()` or :command:`add_library()` but without the
additional option ``GLOBAL`` this is False, too. However, setting this
property for such a locally ``IMPORTED`` target to True promotes that
target to global scope. This promotion can only be done in the same
directory where that ``IMPORTED`` target was created in the first place.

.. note::

  Once an imported target has been made global, it cannot be changed back to
  non-global. Therefore, if a project sets this property, it may only
  provide a value of True. CMake will issue an error if the project tries to
  set the property to a non-True value, even if the value was already False.

.. note::

  Local :ref:`ALIAS targets <Alias Targets>` created before promoting an
  :ref:`IMPORTED target <Imported Targets>` from ``LOCAL`` to ``GLOBAL``, keep
  their initial scope (see :prop_tgt:`ALIAS_GLOBAL` target property).

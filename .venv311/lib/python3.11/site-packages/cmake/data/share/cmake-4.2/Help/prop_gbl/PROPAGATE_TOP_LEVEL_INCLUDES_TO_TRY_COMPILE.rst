PROPAGATE_TOP_LEVEL_INCLUDES_TO_TRY_COMPILE
-------------------------------------------

.. versionadded:: 3.30

When this global property is set to true, the
:variable:`CMAKE_PROJECT_TOP_LEVEL_INCLUDES` variable is propagated into
:command:`try_compile` calls that use the
:ref:`whole-project signature <Try Compiling Whole Projects>`.
Calls to the :ref:`source file signature <Try Compiling Source Files>` are not
affected by this property.
``PROPAGATE_TOP_LEVEL_INCLUDES_TO_TRY_COMPILE`` is unset by default.

For :ref:`dependency providers <dependency_providers_overview>` that want to
be enabled in whole-project :command:`try_compile` calls, set this global
property to true just before or after registering the provider.
Note that all files listed in :variable:`CMAKE_PROJECT_TOP_LEVEL_INCLUDES`
will need to be able to handle being included in such :command:`try_compile`
calls, and it is the user's responsibility to ensure this.

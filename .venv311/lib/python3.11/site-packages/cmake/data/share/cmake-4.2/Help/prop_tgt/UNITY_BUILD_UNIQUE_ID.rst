UNITY_BUILD_UNIQUE_ID
---------------------

.. versionadded:: 3.20

The name of a valid C-identifier which is set to a unique per-file
value during unity builds.

When this property is populated and when :prop_tgt:`UNITY_BUILD`
is true, the property value is used to define a compiler definition
of the specified name. The value of the defined symbol is unspecified,
but it is unique per file path.

Given:

.. code-block:: cmake

  set_target_properties(myTarget PROPERTIES
    UNITY_BUILD "ON"
    UNITY_BUILD_UNIQUE_ID "MY_UNITY_ID"
  )

the ``MY_UNITY_ID`` symbol is defined to a unique per-file value.

One known use case for this identifier is to disambiguate the
variables in an anonymous namespace in a limited scope.
Anonymous namespaces present a problem for unity builds because
they are used to ensure that certain variables and declarations
are scoped to a translation unit which is approximated by a
single source file.  When source files are combined in a unity
build file, those variables in different files are combined in
a single translation unit and the names clash.  This property can
be used to avoid that with code like the following:

.. code-block:: cpp

  // Needed for when unity builds are disabled
  #ifndef MY_UNITY_ID
  #define MY_UNITY_ID
  #endif

  namespace { namespace MY_UNITY_ID {
    // The name 'i' clashes (or could clash) with other
    // variables in other anonymous namespaces
    int i = 42;
  }}

  int use_var()
  {
    return MY_UNITY_ID::i;
  }

The pseudonymous namespace is used within a truly anonymous namespace.
On many platforms, this maintains the invariant that the symbols within
do not get external linkage when performing a unity build.

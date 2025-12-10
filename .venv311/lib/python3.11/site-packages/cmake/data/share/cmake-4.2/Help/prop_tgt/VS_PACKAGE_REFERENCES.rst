VS_PACKAGE_REFERENCES
---------------------

.. versionadded:: 3.15

Visual Studio package references for nuget.

Adds one or more semicolon-delimited package references to a generated
Visual Studio project. The version of the package will be
underscore delimited. For example, ``boost_1.7.0;nunit_3.12.*``.

.. code-block:: cmake

  set_property(TARGET ${TARGET_NAME} PROPERTY
    VS_PACKAGE_REFERENCES "boost_1.7.0")

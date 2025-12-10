CMP0163
-------

.. versionadded:: 3.30

The :prop_sf:`GENERATED` source file property is now visible in all directories.

In CMake 3.29 and below, the :prop_sf:`GENERATED` source file property,
like other source file properties, was scoped in every directory separately.
Although policy :policy:`CMP0118` allowed sources marked ``GENERATED`` in one
directory to be used in other directories without manually marking them as
``GENERATED`` again, the ``GENERATED`` property was still not visible to
:command:`get_property` and :command:`get_source_file_property` calls.

Whether or not a source file is generated is an all-or-nothing global
property of the source: a source is either generated or it is not.
CMake 3.30 and above prefer to treat the :prop_sf:`GENERATED` source file
property as globally scoped.  Once it is set in one directory, it is
immediately visible to :command:`get_property` and
:command:`get_source_file_property` calls in other directories.
This policy provides compatibility for projects that have not been
updated for this behavior.

The ``OLD`` behavior of this policy is for the ``GENERATED`` source file
property to be visible only in the directories in which it is set.  The
``NEW`` behavior of this policy is to allow the ``GENERATED`` source file
property to be visible in all directories once set in any directory.
Furthermore, the ``NEW`` behavior of this policy implies the ``NEW``
behavior of policy :policy:`CMP0118`: the ``GENERATED`` property may
be set only to boolean values, and may not be turned off once turned on.

.. |INTRODUCED_IN_CMAKE_VERSION| replace:: 3.30
.. |WARNS_OR_DOES_NOT_WARN| replace::
   does *not* warn
.. include:: include/STANDARD_ADVICE.rst

.. include:: include/DEPRECATED.rst

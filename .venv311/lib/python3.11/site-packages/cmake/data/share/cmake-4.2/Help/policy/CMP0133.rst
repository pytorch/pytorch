CMP0133
-------

.. versionadded:: 3.24

The :module:`CPack` module disables SLA by default in the
:cpack_gen:`CPack DragNDrop Generator`.

The :cpack_gen:`CPack DragNDrop Generator` in CMake 3.22 and below attach a
Software License Agreement (SLA) to ``.dmg`` files using the file specified
by :variable:`CPACK_RESOURCE_FILE_LICENSE`, if set to a non-default value.
macOS 12.0 deprecated the tools used to do this, so CMake 3.23 added
the :variable:`CPACK_DMG_SLA_USE_RESOURCE_FILE_LICENSE` option to
control the behavior.  CMake 3.23 enables that option by default for
compatibility with older versions. CMake 3.24 and above prefer to *not*
enable the :variable:`CPACK_DMG_SLA_USE_RESOURCE_FILE_LICENSE` option by
default. This policy provides compatibility with projects that have not
been updated to account for the lack of a SLA in their ``.dmg`` packages.

The ``OLD`` behavior for this policy is to enable
:variable:`CPACK_DMG_SLA_USE_RESOURCE_FILE_LICENSE` by default.
The ``NEW`` behavior for this policy is to not enable it by default.

.. |INTRODUCED_IN_CMAKE_VERSION| replace:: 3.24
.. |WARNS_OR_DOES_NOT_WARN| replace:: does *not* warn by default
.. include:: include/STANDARD_ADVICE.rst

See documentation of the
:variable:`CMAKE_POLICY_WARNING_CMP0133 <CMAKE_POLICY_WARNING_CMP<NNNN>>`
variable to control the warning.

.. include:: include/DEPRECATED.rst

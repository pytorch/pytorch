
Note that it is not advisable to populate the :genex:`INSTALL_INTERFACE` of
the |INTERFACE_PROPERTY_LINK| of a target with absolute paths to the include
directories of dependencies.  That would hard-code into installed packages
the include directory paths for dependencies
**as found on the machine the package was made on**.

The :genex:`INSTALL_INTERFACE` of the |INTERFACE_PROPERTY_LINK| is only
suitable for specifying the required include directories for headers
provided with the target itself, not those provided by the transitive
dependencies listed in its :prop_tgt:`INTERFACE_LINK_LIBRARIES` target
property.  Those dependencies should themselves be targets that specify
their own header locations in |INTERFACE_PROPERTY_LINK|.

See the :ref:`Creating Relocatable Packages` section of the
:manual:`cmake-packages(7)` manual for discussion of additional care
that must be taken when specifying usage requirements while creating
packages for redistribution.

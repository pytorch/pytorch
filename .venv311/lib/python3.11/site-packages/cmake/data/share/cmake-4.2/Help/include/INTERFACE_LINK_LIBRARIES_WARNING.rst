
Note that it is not advisable to populate the
|INTERFACE_PROPERTY_LINK| of a target with absolute paths to dependencies.
That would hard-code into installed packages the library file paths
for dependencies **as found on the machine the package was made on**.

See the :ref:`Creating Relocatable Packages` section of the
:manual:`cmake-packages(7)` manual for discussion of additional care
that must be taken when specifying usage requirements while creating
packages for redistribution.

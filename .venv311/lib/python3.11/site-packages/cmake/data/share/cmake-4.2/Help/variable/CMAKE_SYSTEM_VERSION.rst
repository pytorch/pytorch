CMAKE_SYSTEM_VERSION
--------------------

The version of the operating system for which CMake is to build.
See the :variable:`CMAKE_SYSTEM_NAME` variable for the OS name.

System Version for Host Builds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the :variable:`CMAKE_SYSTEM_NAME` variable takes its default value
then ``CMAKE_SYSTEM_VERSION`` is by default set to the same value as the
:variable:`CMAKE_HOST_SYSTEM_VERSION` variable so that the build targets
the host system version.

In the case of a host build then ``CMAKE_SYSTEM_VERSION`` may be set
explicitly when first configuring a new build tree in order to enable
targeting the build for a different version of the host operating system
than is actually running on the host.  This is allowed and not considered
cross compiling so long as the binaries built for the specified OS version
can still run on the host.

System Version for Cross Compiling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the :variable:`CMAKE_SYSTEM_NAME` variable is set explicitly to
enable :ref:`cross compiling <Cross Compiling Toolchain>` then the
value of ``CMAKE_SYSTEM_VERSION`` must also be set explicitly to specify
the target system version.

CMAKE_CROSSCOMPILING
--------------------

This variable is set by CMake to indicate whether it is cross compiling,
but note limitations discussed below.

This variable will be set to true by CMake if the :variable:`CMAKE_SYSTEM_NAME`
variable has been set manually (i.e. in a toolchain file or as a cache entry
from the :manual:`cmake <cmake(1)>` command line). In most cases, manually
setting :variable:`CMAKE_SYSTEM_NAME` will only be done when cross compiling
since, if not manually set, it will be given the same value as
:variable:`CMAKE_HOST_SYSTEM_NAME`, which is correct for
the non-cross-compiling case. In the event that :variable:`CMAKE_SYSTEM_NAME`
is manually set to the same value as :variable:`CMAKE_HOST_SYSTEM_NAME`, then
``CMAKE_CROSSCOMPILING`` will still be set to true.

Another case to be aware of is that builds targeting Apple platforms other than
macOS are handled differently to other cross compiling scenarios. Rather than
relying on :variable:`CMAKE_SYSTEM_NAME` to select the target platform, Apple
device builds use :variable:`CMAKE_OSX_SYSROOT` to select the appropriate SDK,
which indirectly determines the target platform. Furthermore, when using the
:generator:`Xcode` generator, developers can switch between device and
simulator builds at build time rather than having a single
choice at configure time, so the concept
of whether the build is cross compiling or not is more complex. Therefore, the
use of ``CMAKE_CROSSCOMPILING`` is not recommended for projects targeting Apple
devices.

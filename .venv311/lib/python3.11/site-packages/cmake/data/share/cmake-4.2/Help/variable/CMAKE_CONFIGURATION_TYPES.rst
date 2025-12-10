CMAKE_CONFIGURATION_TYPES
-------------------------

Specifies the available build types (configurations) on multi-config
generators (e.g. :ref:`Visual Studio <Visual Studio Generators>`,
:generator:`Xcode`, or :generator:`Ninja Multi-Config`) as a
:ref:`semicolon-separated list <CMake Language Lists>`.  Typical entries
include ``Debug``, ``Release``, ``RelWithDebInfo`` and ``MinSizeRel``,
but custom build types can also be defined.

This variable is initialized by the first :command:`project` or
:command:`enable_language` command called in a project when a new build
tree is first created.  If the :envvar:`CMAKE_CONFIGURATION_TYPES`
environment variable is set, its value is used.  Otherwise, the default
value is generator-specific.

Depending on the situation, the values in this variable may be treated
case-sensitively or case-insensitively.  See :ref:`Build Configurations`
for discussion of this and other related topics.

For single-config generators, see :variable:`CMAKE_BUILD_TYPE`.

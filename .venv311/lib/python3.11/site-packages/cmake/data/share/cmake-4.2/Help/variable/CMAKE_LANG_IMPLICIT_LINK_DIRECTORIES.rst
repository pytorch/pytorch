CMAKE_<LANG>_IMPLICIT_LINK_DIRECTORIES
--------------------------------------

Implicit linker search path detected for language ``<LANG>``.

Compilers typically pass directories containing language runtime
libraries and default library search paths when they invoke a linker.
These paths are implicit linker search directories for the compiler's
language.

For each language enabled by the :command:`project` or
:command:`enable_language` command, CMake automatically detects these
directories and reports the results in this variable.
The :envvar:`CMAKE_<LANG>_IMPLICIT_LINK_DIRECTORIES_EXCLUDE` environment
variable may be set to exclude specific directories from the automatically
detected results.

When linking to a static library, CMake adds the implicit link directories
from this variable for each language used in the static library (except
the language whose compiler is used to drive linking).  In the case of an
imported static library, the :prop_tgt:`IMPORTED_LINK_INTERFACE_LANGUAGES`
target property lists the languages whose implicit link information is
needed.  If any of the languages is not enabled, its value for the
``CMAKE_<LANG>_IMPLICIT_LINK_DIRECTORIES`` variable may instead be provided
by the project.  Or, a :variable:`toolchain file <CMAKE_TOOLCHAIN_FILE>`
may set the variable to a value known for the specified toolchain.  It will
either be overridden when the language is enabled, or used as a fallback.

Some toolchains read implicit directories from an environment variable such as
``LIBRARY_PATH``.  If using such an environment variable, keep its value
consistent when operating in a given build tree because CMake saves the value
detected when first creating a build tree.

In CMake versions prior to 4.0, if policy :policy:`CMP0060` is not set
to ``NEW``, then when a library in one of these directories is given by
full path to :command:`target_link_libraries` CMake will generate the
``-l<name>`` form on link lines for historical purposes.

See also the :variable:`CMAKE_<LANG>_IMPLICIT_LINK_LIBRARIES` variable.

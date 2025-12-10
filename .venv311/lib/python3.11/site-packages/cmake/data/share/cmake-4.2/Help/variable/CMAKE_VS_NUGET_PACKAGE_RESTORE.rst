CMAKE_VS_NUGET_PACKAGE_RESTORE
------------------------------

.. versionadded:: 3.23

When using :ref:`Visual Studio Generators`, this cache variable controls
if msbuild should automatically attempt to restore NuGet packages
prior to a build. NuGet packages can be defined using the
:prop_tgt:`VS_PACKAGE_REFERENCES` property on a target. If no
package references are defined, this setting will do nothing.

The command line option
:option:`--resolve-package-references <cmake--build --resolve-package-references>`
can be used alternatively to control the resolve behavior globally.
This option will take precedence over the cache variable.

Targets that use the :prop_tgt:`DOTNET_SDK` are required to run a
restore before building. Disabling this option may cause the build
to fail in such projects.

This setting is stored as a cache entry. Default value is ``ON``.

See also the :prop_tgt:`VS_PACKAGE_REFERENCES` property.

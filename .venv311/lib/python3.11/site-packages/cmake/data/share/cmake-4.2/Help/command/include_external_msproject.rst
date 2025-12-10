include_external_msproject
--------------------------

Include an external Microsoft project file in the solution file produced
by :ref:`Visual Studio Generators`. Ignored on other generators.

.. code-block:: cmake

  include_external_msproject(projectname location
                             [TYPE projectTypeGUID]
                             [GUID projectGUID]
                             [PLATFORM platformName]
                             dep1 dep2 ...)

Includes an external Microsoft project in the generated solution file.
This will create a target named ``[projectname]``.
This can be used in the :command:`add_dependencies`
command to make things depend on the external project.

``TYPE``, ``GUID`` and ``PLATFORM`` are optional parameters that allow one to
specify the type of project, id (``GUID``) of the project and the name of
the target platform.  This is useful for projects requiring values
other than the default (e.g.  WIX projects).

.. versionadded:: 3.9
  If the imported project has different configuration names than the
  current project, set the :prop_tgt:`MAP_IMPORTED_CONFIG_<CONFIG>`
  target property to specify the mapping.

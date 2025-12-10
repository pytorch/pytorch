VS_STARTUP_PROJECT
------------------

.. versionadded:: 3.6

Specify the default startup project in a Visual Studio solution.

The :ref:`Visual Studio Generators` create a ``.sln`` file for each directory
whose ``CMakeLists.txt`` file calls the :command:`project` command.  Set this
property in the same directory as a :command:`project` command call (e.g. in
the top-level ``CMakeLists.txt`` file) to specify the default startup project
for the corresponding solution file.

The property must be set to the name of an existing target.  This
will cause that project to be listed first in the generated solution
file causing Visual Studio to make it the startup project if the
solution has never been opened before.

If this property is not specified, then the ``ALL_BUILD`` project
will be the default.

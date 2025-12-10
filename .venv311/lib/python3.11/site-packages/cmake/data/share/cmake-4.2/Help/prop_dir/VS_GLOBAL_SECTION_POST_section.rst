VS_GLOBAL_SECTION_POST_<section>
--------------------------------

Specify a postSolution global section in Visual Studio.

Setting a property like this generates an entry of the following form
in the solution file:

::

  GlobalSection(<section>) = postSolution
    <contents based on property value>
  EndGlobalSection

The property must be set to a semicolon-separated list of ``key=value``
pairs.  Each such pair will be transformed into an entry in the
solution global section.  Whitespace around key and value is ignored.
List elements which do not contain an equal sign are skipped.

This property only works for :ref:`Visual Studio Generators`; it is ignored
on other generators.  The property only applies when set on a
directory whose ``CMakeLists.txt`` contains a :command:`project` command.

Note that CMake generates postSolution sections ``ExtensibilityGlobals``
and ``ExtensibilityAddIns`` by default.  If you set the corresponding
property, it will override the default section.  For example, setting
``VS_GLOBAL_SECTION_POST_ExtensibilityGlobals`` will override the default
contents of the ``ExtensibilityGlobals`` section, while keeping
ExtensibilityAddIns on its default.  However, CMake will always
add a ``SolutionGuid`` to the ``ExtensibilityGlobals`` section
if it is not specified explicitly.

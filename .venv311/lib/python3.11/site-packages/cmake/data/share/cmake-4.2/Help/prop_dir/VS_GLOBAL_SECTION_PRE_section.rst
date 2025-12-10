VS_GLOBAL_SECTION_PRE_<section>
-------------------------------

Specify a preSolution global section in Visual Studio.

Setting a property like this generates an entry of the following form
in the solution file:

::

  GlobalSection(<section>) = preSolution
    <contents based on property value>
  EndGlobalSection

The property must be set to a semicolon-separated list of ``key=value``
pairs.  Each such pair will be transformed into an entry in the
solution global section.  Whitespace around key and value is ignored.
List elements which do not contain an equal sign are skipped.

This property only works for :ref:`Visual Studio Generators`; it is ignored
on other generators.  The property only applies when set on a
directory whose ``CMakeLists.txt`` contains a :command:`project` command.

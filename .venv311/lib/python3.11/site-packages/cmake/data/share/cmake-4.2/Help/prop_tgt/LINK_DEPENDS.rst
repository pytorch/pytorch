LINK_DEPENDS
------------

Additional files on which a target binary depends for linking.

Specifies a semicolon-separated list of full-paths to files on which
the link rule for this target depends.  The target binary will be
linked if any of the named files is newer than it.

This property is supported only by :generator:`Ninja` and
:ref:`Makefile Generators`.  It is
intended to specify dependencies on "linker scripts" for custom Makefile link
rules.

Contents of ``LINK_DEPENDS`` may use "generator expressions" with
the syntax ``$<...>``.  See the :manual:`cmake-generator-expressions(7)`
manual for available expressions.  See the :manual:`cmake-buildsystem(7)`
manual for more on defining buildsystem properties.

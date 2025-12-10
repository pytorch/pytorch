LINK_INTERFACE_MULTIPLICITY
---------------------------

Repetition count for ``STATIC`` libraries with cyclic dependencies.

When linking to a ``STATIC`` library target with cyclic dependencies the
linker may need to scan more than once through the archives in the
strongly connected component of the dependency graph.  CMake by
default constructs the link line so that the linker will scan through
the component at least twice.  This property specifies the minimum
number of scans if it is larger than the default.  CMake uses the
largest value specified by any target in a component.

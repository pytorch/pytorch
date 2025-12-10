OSX_ARCHITECTURES
-----------------

Target specific architectures for macOS.

The ``OSX_ARCHITECTURES`` property sets the target binary architecture for
targets on macOS (``-arch``).  This property is initialized by the value of the
variable :variable:`CMAKE_OSX_ARCHITECTURES` if it is set when a target is
created.  Use :prop_tgt:`OSX_ARCHITECTURES_<CONFIG>` to set the binary
architectures on a per-configuration basis, where ``<CONFIG>`` is an
upper-case name (e.g. ``OSX_ARCHITECTURES_DEBUG``).

Indicate to :ref:`Visual Studio Generators` what configurations are considered
debug configurations.  This controls the ``UseDebugLibraries`` setting in
each configuration of a ``.vcxproj`` file.

The "Use Debug Libraries" setting in Visual Studio projects, despite its
specific-sounding name, is a general-purpose indicator of what configurations
are considered debug configurations.  In standalone projects, this may affect
MSBuild's default selection of MSVC runtime library, optimization flags,
runtime checks, and similar settings.  In CMake projects those settings are
typically generated explicitly based on the project's specification, e.g.,
the MSVC runtime library is controlled by |MSVC_RUNTIME_LIBRARY|.  However,
the ``UseDebugLibraries`` indicator is useful for reference by both humans
and tools, and may also affect the behavior of platform-specific SDKs.

Set |VS_USE_DEBUG_LIBRARIES| to a true or false value to indicate whether
each configuration is considered a debug configuration.  The value may also
be the empty string (``""``) in which case no ``UseDebugLibraries`` will be
added explicitly by CMake, and MSBuild will use its default value, ``false``.

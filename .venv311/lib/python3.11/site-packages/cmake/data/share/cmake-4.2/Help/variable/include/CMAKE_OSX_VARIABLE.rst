The value of this variable should be set prior to the first
:command:`project` or :command:`enable_language` command invocation
because it may influence configuration of the toolchain and flags.
It is intended to be set locally by the user creating a build tree.
This variable should be set as a ``CACHE`` entry (or else CMake may
remove it while initializing a cache entry of the same name) unless
policy :policy:`CMP0126` is set to ``NEW``.

Despite the ``OSX`` part in the variable name(s) they apply also to
other SDKs than macOS like iOS, tvOS, visionOS, or watchOS.

This variable is ignored on platforms other than Apple.

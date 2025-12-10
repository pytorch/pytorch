CMAKE_WARN_DEPRECATED
---------------------

Whether to issue warnings for deprecated functionality.

If not ``FALSE``, use of deprecated functionality will issue warnings.
If this variable is not set, CMake behaves as if it were set to ``TRUE``.

When running :manual:`cmake(1)`, this option can be enabled with the
:option:`-Wdeprecated <cmake -Wdeprecated>` option, or disabled with the
:option:`-Wno-deprecated <cmake -Wno-deprecated>` option.

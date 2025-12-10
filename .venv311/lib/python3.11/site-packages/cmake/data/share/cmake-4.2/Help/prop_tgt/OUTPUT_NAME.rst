OUTPUT_NAME
-----------

Output name for target files.

This sets the base name for output files created for an executable or
library target.  If not set, the logical target name is used by
default during generation. The value is not set by default during
configuration.

Contents of ``OUTPUT_NAME`` and the variants listed below may use
:manual:`generator expressions <cmake-generator-expressions(7)>`.

See also the variants:

* :prop_tgt:`OUTPUT_NAME_<CONFIG>`
* :prop_tgt:`ARCHIVE_OUTPUT_NAME_<CONFIG>`
* :prop_tgt:`ARCHIVE_OUTPUT_NAME`
* :prop_tgt:`LIBRARY_OUTPUT_NAME_<CONFIG>`
* :prop_tgt:`LIBRARY_OUTPUT_NAME`
* :prop_tgt:`RUNTIME_OUTPUT_NAME_<CONFIG>`
* :prop_tgt:`RUNTIME_OUTPUT_NAME`

# API docs

This document describes the conventions behind the THNN API.

### The API

All functions provided by THNN are stored in `aten/src/THNN/generic/THNN.h`.
Look at this file.

### Note on function names

Please remember, that because C doesn't support function overloading, functions taking different tensor types have different names. So e.g. for an Abs module, there are actually two updateOutput functions:

* `void THNN_FloatAbs_updateOutput(...)`
* `void THNN_DoubleAbs_updateOutput(...)`

In these docs such function will be referred to as `void THNN_Abs_updateOutput(...)`, and it's up to developer to add a type prefix. `real` is an alias for that type.

### Argument types

Some arguments have additional tags placed in square brackets in their header declarations:

* **[OUT]** - This is the output argument. It will be reshaped if needed.
* **[OPTIONAL]** - This argument is optional and can be safely set to NULL
* **[BUFFER]** - A buffer. `updateGradInput` and `accGradParameters` should get the same buffers that were used in `updateOutput` call.
* **[MODIFIED]** - Some functions accept an `inplace` flag. If set to true, this argument might be modified (in addition to the output).


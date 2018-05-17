All files living in this directory are written with the assumption that MIOpen is available,
which means that these code are not guarded by `#if AT_MIOPEN_ENABLED()`. Therefore, whenever
you need to use definitions from here, please guard the `#include<ATen/miopen/*.h>` and
definition usages with `#if AT_MIOPEN_ENABLED()` macro, e.g. [native/miopen/BatchNorm.cpp](native/miopen/BatchNorm.cpp).

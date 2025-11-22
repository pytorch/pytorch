#pragma clang diagnostic push
// ExecuTorch depends on these files and is pinned to C++17
#pragma clang diagnostic error "-Wpre-c++20-compat"

#include <torch/headeronly/util/Half.h>

// need to keep the following for BC because the APIs in here were exposed
// before migrating Half to torch/headeronly
#if (defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)) && \
    !defined(__APPLE__)
#include <ATen/cpu/vec/vec_half.h>
#endif

#pragma clang diagnostic pop

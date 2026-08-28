#include <torch/headeronly/util/Half.h>

// need to keep the following for BC because the APIs in here were exposed
// before migrating Half to torch/headeronly
#if (defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)) && \
    !defined(__APPLE__)
#include <ATen/cpu/vec/vec_half.h>
#endif

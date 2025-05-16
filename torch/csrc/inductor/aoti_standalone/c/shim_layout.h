#pragma once

// This header mimics APIs in aoti_torch/c/shim.h in a standalone way

// TODO: Move Layout to a header-only directory
#include <c10/core/Layout.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AOTI_LAYOUT_IMPL(layout_str, layout_type)   \
  inline int32_t aoti_torch_layout_##layout_str() { \
    return (int32_t)c10::Layout::layout_type;       \
  }

AOTI_LAYOUT_IMPL(strided, Strided)
AOTI_LAYOUT_IMPL(sparse, Sparse)
AOTI_LAYOUT_IMPL(sparse_csr, SparseCsr)
AOTI_LAYOUT_IMPL(mkldnn, Mkldnn)
AOTI_LAYOUT_IMPL(sparse_csc, SparseCsc)
AOTI_LAYOUT_IMPL(sparse_bsr, SparseBsr)
AOTI_LAYOUT_IMPL(sparse_bsc, SparseBsc)
AOTI_LAYOUT_IMPL(jagged, Jagged)
#undef AOTI_LAYOUT_IMPL

#ifdef __cplusplus
} // extern "C"
#endif

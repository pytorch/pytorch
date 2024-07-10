#define KERNEL_NAME pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2
#define W_INDEX_DTYPE uint32_t
#include "8x4c1x4-dq-packedA-sse2.h"
#undef KERNEL_NAME
#undef W_INDEX_DTYPE

#define KERNEL_NAME pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2
#define W_INDEX_DTYPE uint16_t
#include "8x4c1x4-dq-packedA-sse2.h"
#undef KERNEL_NAME
#undef W_INDEX_DTYPE

#define KERNEL_NAME pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2
#define W_INDEX_DTYPE uint8_t
#include "8x4c1x4-dq-packedA-sse2.h"
#undef KERNEL_NAME
#undef W_INDEX_DTYPE

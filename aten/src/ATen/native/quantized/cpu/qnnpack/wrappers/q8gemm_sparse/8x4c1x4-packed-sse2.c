#if defined(__i386__) || defined(__i686__) || defined(__x86_64__)
#include <q8gemm_sparse/8x4-packA-sse2.c>
#include <q8gemm_sparse/8x4c1x4-dq-packedA-sse2.c>
#endif /* defined(__i386__) || defined(__i686__) || defined(__x86_64__) */

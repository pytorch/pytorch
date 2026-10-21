#pragma once

#ifdef USE_ROCM

#define AOTRITON_VERSION_INT(x, y) (x * 100 + y)
#define AOTRITON_VERSION_CURRENT (AOTRITON_VERSION_MAJOR * 100 + AOTRITON_VERSION_MINOR)

#if AOTRITON_VERSION_CURRENT >= AOTRITON_VERSION_INT(0, 11)
#define AOTRITON_ALWAYS_V3_API 1
#else
#define AOTRITON_ALWAYS_V3_API 0
#endif

#if AOTRITON_VERSION_CURRENT >= AOTRITON_VERSION_INT(0, 10)
#define AOTRITON_V3_API 1
#else
#define AOTRITON_V3_API 0
#endif

#if AOTRITON_VERSION_CURRENT >= AOTRITON_VERSION_INT(0, 12)
#define AOTRITON_COMPACT_VARLEN_LSE 1
#else
#define AOTRITON_COMPACT_VARLEN_LSE 1
#endif

// AOTriton 0.14 replaced attn_{fwd,bwd}_params's VarlenType enum + cu_seqlens_q/k
// (+ seq_strides_q/k) fields with a VarlenBits bit-field struct + seqinfo_q/k0
// (+ seqinfo_q/k1) fields. Neither spelling exists in the other version's header,
// so callers must pick one at compile time.
#if AOTRITON_VERSION_CURRENT >= AOTRITON_VERSION_INT(0, 14)
#define AOTRITON_VARLEN_BITS_API 1
#else
#define AOTRITON_VARLEN_BITS_API 0
#endif

#endif

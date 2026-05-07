#pragma once

// Test these using #if AT_CUDNN_ENABLED(), not #ifdef, so that it's
// obvious if you forgot to include Config.h
//    c.f. https://stackoverflow.com/questions/33759787/generating-an-error-if-checked-boolean-macro-is-not-defined
//
// NB: This header MUST NOT be included from other headers; it should
// only be included from C++ files.
#define AT_CUDNN_ENABLED() 1
#define AT_CUSPARSELT_ENABLED() 1
#define AT_HIPSPARSELT_ENABLED() 1
#define AT_ROCM_ENABLED() 1
#define AT_MAGMA_ENABLED() 1

// Needed for hipMAGMA to correctly identify implementation
#if (AT_ROCM_ENABLED() && AT_MAGMA_ENABLED())
#define HAVE_HIP 1
#endif

#define NVCC_FLAGS_EXTRA "-gencode;arch=compute_75,code=sm_75"

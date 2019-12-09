#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathMagma.h"
#else

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)

// MAGMA (i.e. CUDA implementation of LAPACK functions)
THC_API void THCTensor_(gels)(THCState *state, THCTensor *rb_, THCTensor *ra_, THCTensor *b_, THCTensor *a_);
THC_API void THCTensor_(geev)(THCState *state, THCTensor *re_, THCTensor *rv_, THCTensor *a_, bool eigenvectors);
THC_API void THCTensor_(potri)(THCState *state, THCTensor *ra_, THCTensor *a, bool upper);
THC_API void THCTensor_(geqrf)(THCState *state, THCTensor *ra_, THCTensor *rtau_, THCTensor *a_);

#endif // defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)

#endif

#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathPointwise.h"
#else

#if !defined(THC_REAL_IS_BOOL)

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)

THC_API void THCTensor_(atan)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(sqrt)(THCState *state, THCTensor *self, THCTensor *src);

#endif

THC_API void THCTensor_(clamp)(THCState *state, THCTensor *self, THCTensor *src, scalar_t min_value, scalar_t max_value);
THC_API void THCTensor_(crossKernel)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2, int dimension);

THC_API void THCTensor_(cadd)(THCState *state, THCTensor *self, THCTensor *src1, scalar_t value, THCTensor *src2);
THC_API void THCTensor_(csub)(THCState *state, THCTensor *self, THCTensor *src1, scalar_t value, THCTensor *src2);
THC_API void THCTensor_(cmul)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(cdiv)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(clshift)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(crshift)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(cfmod)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2);

#endif
#endif

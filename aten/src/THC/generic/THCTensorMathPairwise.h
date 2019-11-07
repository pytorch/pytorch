#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathPairwise.h"
#else

THC_API int THCTensor_(equal)(THCState *state, THCTensor *self, THCTensor *src);

THC_API void THCTensor_(bitand)(THCState *state, THCTensor *self, THCTensor *src, scalar_t value);
THC_API void THCTensor_(bitor)(THCState *state, THCTensor *self, THCTensor *src, scalar_t value);
THC_API void THCTensor_(bitxor)(THCState *state, THCTensor *self, THCTensor *src, scalar_t value);

#if !defined(THC_REAL_IS_BOOL)

THC_API void THCTensor_(mul)(THCState *state, THCTensor *self, THCTensor *src, scalar_t value);
THC_API void THCTensor_(div)(THCState *state, THCTensor *self, THCTensor *src, scalar_t value);
THC_API void THCTensor_(lshift)(THCState *state, THCTensor *self, THCTensor *src, scalar_t value);
THC_API void THCTensor_(rshift)(THCState *state, THCTensor *self, THCTensor *src, scalar_t value);
THC_API void THCTensor_(fmod)(THCState *state, THCTensor *self, THCTensor *src, scalar_t value);
THC_API void THCTensor_(remainder)(THCState *state, THCTensor *self, THCTensor *src, scalar_t value);

#endif

#endif

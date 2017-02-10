#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathPairwise.h"
#else

THC_API void THCTensor_(add)(THCState *state, THCTensor *self, THCTensor *src, real value);
THC_API void THCTensor_(sub)(THCState *state, THCTensor *self, THCTensor *src, real value);
THC_API void THCTensor_(mul)(THCState *state, THCTensor *self, THCTensor *src, real value);
THC_API void THCTensor_(div)(THCState *state, THCTensor *self, THCTensor *src, real value);
THC_API void THCTensor_(lshift)(THCState *state, THCTensor *self, THCTensor *src, real value);
THC_API void THCTensor_(rshift)(THCState *state, THCTensor *self, THCTensor *src, real value);
THC_API void THCTensor_(fmod)(THCState *state, THCTensor *self, THCTensor *src, real value);
THC_API void THCTensor_(remainder)(THCState *state, THCTensor *self, THCTensor *src, real value);
THC_API void THCTensor_(bitand)(THCState *state, THCTensor *self, THCTensor *src, real value);
THC_API void THCTensor_(bitor)(THCState *state, THCTensor *self, THCTensor *src, real value);
THC_API void THCTensor_(bitxor)(THCState *state, THCTensor *self, THCTensor *src, real value);

THC_API int THCTensor_(equal)(THCState *state, THCTensor *self, THCTensor *src);

#endif

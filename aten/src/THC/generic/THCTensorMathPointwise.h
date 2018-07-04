#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathPointwise.h"
#else

THC_API void THCTensor_(pow)(THCState *state, THCTensor *self, THCTensor *src, real value);
THC_API void THCTensor_(tpow)(THCState *state, THCTensor *self, real value, THCTensor *src);
THC_API void THCTensor_(cpow)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2);

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)

THC_API void THCTensor_(sigmoid)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(log)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(lgamma)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(digamma)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(polygamma)(THCState *state, THCTensor *self, int64_t n, THCTensor *src);
THC_API void THCTensor_(log10)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(log1p)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(log2)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(exp)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(expm1)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(cos)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(acos)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(cosh)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(sin)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(asin)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(sinh)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(tan)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(atan)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(atan2)(THCState *state, THCTensor *r_, THCTensor *tx, THCTensor *ty);
THC_API void THCTensor_(tanh)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(erf)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(erfinv)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(sqrt)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(rsqrt)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(ceil)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(floor)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(round)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(trunc)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(frac)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(lerp)(THCState *state, THCTensor *result, THCTensor *a, THCTensor *b, real w);

THC_API void THCTensor_(cinv)(THCState *state, THCTensor *self, THCTensor *src);

#endif

THC_API void THCTensor_(neg)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(abs)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(sign)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(clamp)(THCState *state, THCTensor *self, THCTensor *src, real min_value, real max_value);
THC_API void THCTensor_(cross)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2, int dimension);

THC_API void THCTensor_(cadd)(THCState *state, THCTensor *self, THCTensor *src1, real value, THCTensor *src2);
THC_API void THCTensor_(csub)(THCState *state, THCTensor *self, THCTensor *src1, real value, THCTensor *src2);
THC_API void THCTensor_(cmul)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(cdiv)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(clshift)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(crshift)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(cmax)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(cmin)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(cfmod)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(cremainder)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(cmaxValue)(THCState *state, THCTensor *self, THCTensor *src, real value);
THC_API void THCTensor_(cminValue)(THCState *state, THCTensor *self, THCTensor *src, real value);
THC_API void THCTensor_(cbitand)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(cbitor)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(cbitxor)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2);

THC_API void THCTensor_(addcmul)(THCState *state, THCTensor *self, THCTensor* t, real value, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(addcdiv)(THCState *state, THCTensor *self, THCTensor* t, real value, THCTensor *src1, THCTensor *src2);

#endif

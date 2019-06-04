#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorMath.h"
#else

#include <ATen/core/Generator.h>
#include <ATen/core/DistributionsHelper.h>

TH_API void THTensor_(nonzero)(THLongTensor *subscript, THTensor *tensor);

#ifndef TH_REAL_IS_HALF

TH_API void THTensor_(ltValue)(THByteTensor *r_, THTensor* t, scalar_t value);
TH_API void THTensor_(leValue)(THByteTensor *r_, THTensor* t, scalar_t value);
TH_API void THTensor_(gtValue)(THByteTensor *r_, THTensor* t, scalar_t value);
TH_API void THTensor_(geValue)(THByteTensor *r_, THTensor* t, scalar_t value);
TH_API void THTensor_(neValue)(THByteTensor *r_, THTensor* t, scalar_t value);
TH_API void THTensor_(eqValue)(THByteTensor *r_, THTensor* t, scalar_t value);

TH_API void THTensor_(ltValueT)(THTensor *r_, THTensor* t, scalar_t value);
TH_API void THTensor_(leValueT)(THTensor *r_, THTensor* t, scalar_t value);
TH_API void THTensor_(gtValueT)(THTensor *r_, THTensor* t, scalar_t value);
TH_API void THTensor_(geValueT)(THTensor *r_, THTensor* t, scalar_t value);
TH_API void THTensor_(neValueT)(THTensor *r_, THTensor* t, scalar_t value);
TH_API void THTensor_(eqValueT)(THTensor *r_, THTensor* t, scalar_t value);

TH_API void THTensor_(ltTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
TH_API void THTensor_(leTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
TH_API void THTensor_(gtTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
TH_API void THTensor_(geTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
TH_API void THTensor_(neTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
TH_API void THTensor_(eqTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);

TH_API void THTensor_(ltTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
TH_API void THTensor_(leTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
TH_API void THTensor_(gtTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
TH_API void THTensor_(geTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
TH_API void THTensor_(neTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);
TH_API void THTensor_(eqTensorT)(THTensor *r_, THTensor *ta, THTensor *tb);

TH_API accreal THTensor_(sumall)(THTensor *t);
TH_API int THTensor_(equal)(THTensor *ta, THTensor *tb);

TH_API void THTensor_(maskedSelect)(THTensor *tensor, THTensor* src, THByteTensor *mask);
TH_API void THTensor_(maskedSelectBool)(THTensor *tensor, THTensor* src, THBoolTensor *mask);

TH_API void THTensor_(bitand)(THTensor *r_, THTensor *t, scalar_t value);
TH_API void THTensor_(cbitand)(THTensor *r_, THTensor *t, THTensor *src);
TH_API void THTensor_(bitor)(THTensor *r_, THTensor *t, scalar_t value);
TH_API void THTensor_(cbitor)(THTensor *r_, THTensor *t, THTensor *src);
TH_API void THTensor_(bitxor)(THTensor *r_, THTensor *t, scalar_t value);
TH_API void THTensor_(cbitxor)(THTensor *r_, THTensor *t, THTensor *src);

TH_API void THTensor_(sign)(THTensor *r_, THTensor *t);

TH_API ptrdiff_t THTensor_(numel)(THTensor *t);
void THTensor_(preserveReduceDimSemantics)(THTensor *r_, int in_dims, int reduce_dimension, int keepdim);

TH_API void THTensor_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
TH_API void THTensor_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
TH_API scalar_t THTensor_(minall)(THTensor *t);
TH_API scalar_t THTensor_(maxall)(THTensor *t);

TH_API void THTensor_(cmax)(THTensor *r, THTensor *t, THTensor *src);
TH_API void THTensor_(cmin)(THTensor *r, THTensor *t, THTensor *src);
TH_API void THTensor_(cmaxValue)(THTensor *r, THTensor *t, scalar_t value);
TH_API void THTensor_(cminValue)(THTensor *r, THTensor *t, scalar_t value);

#if !defined(TH_REAL_IS_BOOL) /* non bool only part */

TH_API void THTensor_(maskedFill)(THTensor *tensor, THByteTensor *mask, scalar_t value);
TH_API void THTensor_(maskedCopy)(THTensor *tensor, THByteTensor *mask, THTensor* src);
TH_API void THTensor_(maskedFillBool)(THTensor *tensor, THBoolTensor *mask, scalar_t value);
TH_API void THTensor_(maskedCopyBool)(THTensor *tensor, THBoolTensor *mask, THTensor* src);

TH_API void THTensor_(indexSelect)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index);
TH_API void THTensor_(indexCopy)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
TH_API void THTensor_(indexAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
TH_API void THTensor_(indexFill)(THTensor *tensor, int dim, THLongTensor *index, scalar_t val);
TH_API void THTensor_(take)(THTensor *tensor, THTensor *src, THLongTensor *index);
TH_API void THTensor_(put)(THTensor *tensor, THLongTensor *index, THTensor *src, int accumulate);

TH_API void THTensor_(gather)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index);
TH_API void THTensor_(scatter)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
TH_API void THTensor_(scatterAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src);
TH_API void THTensor_(scatterFill)(THTensor *tensor, int dim, THLongTensor *index, scalar_t val);

TH_API accreal THTensor_(dot)(THTensor *t, THTensor *src);

TH_API void THTensor_(neg)(THTensor *self, THTensor *src);
TH_API void THTensor_(cinv)(THTensor *self, THTensor *src);

TH_API void THTensor_(add)(THTensor *r_, THTensor *t, scalar_t value);
TH_API void THTensor_(sub)(THTensor *r_, THTensor *t, scalar_t value);
TH_API void THTensor_(add_scaled)(THTensor *r_, THTensor *t, scalar_t value, scalar_t alpha);
TH_API void THTensor_(sub_scaled)(THTensor *r_, THTensor *t, scalar_t value, scalar_t alpha);
TH_API void THTensor_(mul)(THTensor *r_, THTensor *t, scalar_t value);
TH_API void THTensor_(div)(THTensor *r_, THTensor *t, scalar_t value);
TH_API void THTensor_(lshift)(THTensor *r_, THTensor *t, scalar_t value);
TH_API void THTensor_(rshift)(THTensor *r_, THTensor *t, scalar_t value);
TH_API void THTensor_(fmod)(THTensor *r_, THTensor *t, scalar_t value);
TH_API void THTensor_(remainder)(THTensor *r_, THTensor *t, scalar_t value);
TH_API void THTensor_(clamp)(THTensor *r_, THTensor *t, scalar_t min_value, scalar_t max_value);

TH_API void THTensor_(cadd)(THTensor *r_, THTensor *t, scalar_t value, THTensor *src);
TH_API void THTensor_(csub)(THTensor *self, THTensor *src1, scalar_t value, THTensor *src2);
TH_API void THTensor_(cmul)(THTensor *r_, THTensor *t, THTensor *src);
TH_API void THTensor_(cpow)(THTensor *r_, THTensor *t, THTensor *src);
TH_API void THTensor_(cdiv)(THTensor *r_, THTensor *t, THTensor *src);
TH_API void THTensor_(clshift)(THTensor *r_, THTensor *t, THTensor *src);
TH_API void THTensor_(crshift)(THTensor *r_, THTensor *t, THTensor *src);
TH_API void THTensor_(cfmod)(THTensor *r_, THTensor *t, THTensor *src);
TH_API void THTensor_(cremainder)(THTensor *r_, THTensor *t, THTensor *src);

TH_API void THTensor_(addcmul)(THTensor *r_, THTensor *t, scalar_t value, THTensor *src1, THTensor *src2);
TH_API void THTensor_(addcdiv)(THTensor *r_, THTensor *t, scalar_t value, THTensor *src1, THTensor *src2);

TH_API void THTensor_(addmv)(THTensor *r_, scalar_t beta, THTensor *t, scalar_t alpha, THTensor *mat,  THTensor *vec);
TH_API void THTensor_(addmm)(THTensor *r_, scalar_t beta, THTensor *t, scalar_t alpha, THTensor *mat1, THTensor *mat2);
TH_API void THTensor_(addr)(THTensor *r_,  scalar_t beta, THTensor *t, scalar_t alpha, THTensor *vec1, THTensor *vec2);

TH_API void THTensor_(addbmm)(THTensor *r_, scalar_t beta, THTensor *t, scalar_t alpha, THTensor *batch1, THTensor *batch2);
TH_API void THTensor_(baddbmm)(THTensor *r_, scalar_t beta, THTensor *t, scalar_t alpha, THTensor *batch1, THTensor *batch2);

TH_API void THTensor_(match)(THTensor *r_, THTensor *m1, THTensor *m2, scalar_t gain);

TH_API void THTensor_(kthvalue)(THTensor *values_, THLongTensor *indices_, THTensor *t, int64_t k, int dimension, int keepdim);
TH_API void THTensor_(mode)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
TH_API void THTensor_(prod)(THTensor *r_, THTensor *t, int dimension, int keepdim);
TH_API void THTensor_(cumsum)(THTensor *r_, THTensor *t, int dimension);
TH_API void THTensor_(cumprod)(THTensor *r_, THTensor *t, int dimension);
TH_API accreal THTensor_(trace)(THTensor *t);

TH_API void THTensor_(diag)(THTensor *r_, THTensor *t, int k);

TH_API void THTensor_(sort)(THTensor *rt_, THLongTensor *ri_, THTensor *t, int dimension, int descendingOrder);
TH_API void THTensor_(topk)(THTensor *rt_, THLongTensor *ri_, THTensor *t, int64_t k, int dim, int dir, int sorted);
TH_API void THTensor_(triu)(THTensor *r_, THTensor *t, int64_t k);

TH_API void THTensor_(pow)(THTensor *r_, THTensor *t, scalar_t value);
TH_API void THTensor_(tpow)(THTensor *r_, scalar_t value, THTensor *t);
TH_API void THTensor_(abs)(THTensor *r_, THTensor *t);

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

TH_API void THTensor_(sigmoid)(THTensor *r_, THTensor *t);
TH_API void THTensor_(log)(THTensor *r_, THTensor *t);
TH_API void THTensor_(lgamma)(THTensor *r_, THTensor *t);
TH_API void THTensor_(digamma)(THTensor *r_, THTensor *t);
TH_API void THTensor_(trigamma)(THTensor *r_, THTensor *t);
TH_API void THTensor_(polygamma)(THTensor *r_, int64_t n, THTensor *t);
TH_API void THTensor_(log10)(THTensor *r_, THTensor *t);
TH_API void THTensor_(log1p)(THTensor *r_, THTensor *t);
TH_API void THTensor_(log2)(THTensor *r_, THTensor *t);
TH_API void THTensor_(exp)(THTensor *r_, THTensor *t);
TH_API void THTensor_(expm1)(THTensor *r_, THTensor *t);
TH_API void THTensor_(cos)(THTensor *r_, THTensor *t);
TH_API void THTensor_(acos)(THTensor *r_, THTensor *t);
TH_API void THTensor_(cosh)(THTensor *r_, THTensor *t);
TH_API void THTensor_(sin)(THTensor *r_, THTensor *t);
TH_API void THTensor_(asin)(THTensor *r_, THTensor *t);
TH_API void THTensor_(sinh)(THTensor *r_, THTensor *t);
TH_API void THTensor_(tan)(THTensor *r_, THTensor *t);
TH_API void THTensor_(atan)(THTensor *r_, THTensor *t);
TH_API void THTensor_(atan2)(THTensor *r_, THTensor *tx, THTensor *ty);
TH_API void THTensor_(tanh)(THTensor *r_, THTensor *t);
TH_API void THTensor_(erf)(THTensor *r_, THTensor *t);
TH_API void THTensor_(erfc)(THTensor *r_, THTensor *t);
TH_API void THTensor_(erfinv)(THTensor *r_, THTensor *t);
TH_API void THTensor_(sqrt)(THTensor *r_, THTensor *t);
TH_API void THTensor_(rsqrt)(THTensor *r_, THTensor *t);
TH_API void THTensor_(ceil)(THTensor *r_, THTensor *t);
TH_API void THTensor_(floor)(THTensor *r_, THTensor *t);
TH_API void THTensor_(round)(THTensor *r_, THTensor *t);
TH_API void THTensor_(trunc)(THTensor *r_, THTensor *t);
TH_API void THTensor_(frac)(THTensor *r_, THTensor *t);

TH_API void THTensor_(std)(THTensor *r_, THTensor *t, int dimension, int biased, int keepdim);
TH_API void THTensor_(var)(THTensor *r_, THTensor *t, int dimension, int biased, int keepdim);
TH_API void THTensor_(norm)(THTensor *r_, THTensor *t, scalar_t value, int dimension, int keepdim);
TH_API void THTensor_(renorm)(THTensor *r_, THTensor *t, scalar_t value, int dimension, scalar_t maxnorm);
TH_API accreal THTensor_(dist)(THTensor *a, THTensor *b, scalar_t value);
TH_API void THTensor_(histc)(THTensor *hist, THTensor *tensor, int64_t nbins, scalar_t minvalue, scalar_t maxvalue);
TH_API void THTensor_(bhistc)(THTensor *hist, THTensor *tensor, int64_t nbins, scalar_t minvalue, scalar_t maxvalue);

TH_API accreal THTensor_(meanall)(THTensor *self);
TH_API accreal THTensor_(varall)(THTensor *self, int biased);
TH_API accreal THTensor_(stdall)(THTensor *self, int biased);
TH_API accreal THTensor_(normall)(THTensor *t, scalar_t value);

TH_API void THTensor_(dirichlet_grad)(THTensor *self, THTensor *x, THTensor *alpha, THTensor *total);
#endif

#endif
#endif
#endif

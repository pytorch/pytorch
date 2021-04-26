#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorMath.h"
#else

#include <ATen/core/Generator.h>

TH_API void THTensor_(nonzero)(THLongTensor *subscript, THTensor *tensor);
TH_API int THTensor_(equal)(THTensor *ta, THTensor *tb);

#if !defined(TH_REAL_IS_HALF)

TH_API ptrdiff_t THTensor_(numel)(THTensor *t);

#if !defined(TH_REAL_IS_BFLOAT16)

void THTensor_(preserveReduceDimSemantics)(THTensor *r_, int in_dims, int reduce_dimension, int keepdim);

TH_API void THTensor_(take)(THTensor *tensor, THTensor *src, THLongTensor *index);
TH_API void THTensor_(put)(THTensor *tensor, THLongTensor *index, THTensor *src, int accumulate);

#if !defined(TH_REAL_IS_BOOL) /* non bool only part */

TH_API void THTensor_(kthvalue)(THTensor *values_, THLongTensor *indices_, THTensor *t, int64_t k, int dimension, int keepdim);

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

TH_API void THTensor_(renorm)(THTensor *r_, THTensor *t, scalar_t value, int dimension, scalar_t maxnorm);
TH_API void THTensor_(histc)(THTensor *hist, THTensor *tensor, int64_t nbins, scalar_t minvalue, scalar_t maxvalue);

TH_API accreal THTensor_(var_all)(THTensor *self, bool unbiased);
TH_API accreal THTensor_(std_all)(THTensor *self, bool unbiased);

#endif
#endif
#endif
#endif /* !defined(TH_REAL_IS_HALF) */
#endif /* TH_GENERIC_FILE*/

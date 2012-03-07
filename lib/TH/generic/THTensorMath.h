#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorMath.h"
#else

TH_API void THTensor_(fill)(THTensor *r_, real value);
TH_API void THTensor_(zero)(THTensor *r_);

TH_API void THTensor_(maskedFill)(THTensor *tensor, THByteTensor *mask, real value);
TH_API void THTensor_(maskedCopy)(THTensor *tensor, THByteTensor *mask, THTensor* src);
TH_API void THTensor_(maskedSelect)(THTensor *tensor, THTensor* src, THByteTensor *mask);

TH_API accreal THTensor_(dot)(THTensor *t, THTensor *src);
  
TH_API real THTensor_(minall)(THTensor *t);
TH_API real THTensor_(maxall)(THTensor *t);
TH_API accreal THTensor_(sumall)(THTensor *t);

TH_API void THTensor_(add)(THTensor *r_, THTensor *t, real value);
TH_API void THTensor_(mul)(THTensor *r_, THTensor *t, real value);
TH_API void THTensor_(div)(THTensor *r_, THTensor *t, real value);

TH_API void THTensor_(cadd)(THTensor *r_, THTensor *t, real value, THTensor *src);  
TH_API void THTensor_(cmul)(THTensor *r_, THTensor *t, THTensor *src);
TH_API void THTensor_(cdiv)(THTensor *r_, THTensor *t, THTensor *src);

TH_API void THTensor_(addcmul)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2);
TH_API void THTensor_(addcdiv)(THTensor *r_, THTensor *t, real value, THTensor *src1, THTensor *src2);

TH_API void THTensor_(addmv)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat,  THTensor *vec);
TH_API void THTensor_(addmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat1, THTensor *mat2);
TH_API void THTensor_(addr)(THTensor *r_,  real beta, THTensor *t, real alpha, THTensor *vec1, THTensor *vec2);

TH_API long THTensor_(numel)(THTensor *t);
TH_API void THTensor_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension);
TH_API void THTensor_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension);
TH_API void THTensor_(sum)(THTensor *r_, THTensor *t, int dimension);
TH_API void THTensor_(prod)(THTensor *r_, THTensor *t, int dimension);
TH_API void THTensor_(cumsum)(THTensor *r_, THTensor *t, int dimension);
TH_API void THTensor_(cumprod)(THTensor *r_, THTensor *t, int dimension);
TH_API void THTensor_(sign)(THTensor *r_, THTensor *t);
TH_API accreal THTensor_(trace)(THTensor *t);
TH_API void THTensor_(cross)(THTensor *r_, THTensor *a, THTensor *b, int dimension);

TH_API void THTensor_(zeros)(THTensor *r_, THLongStorage *size);
TH_API void THTensor_(ones)(THTensor *r_, THLongStorage *size);
TH_API void THTensor_(diag)(THTensor *r_, THTensor *t, int k);
TH_API void THTensor_(eye)(THTensor *r_, long n, long m);
TH_API void THTensor_(range)(THTensor *r_, real xmin, real xmax, real step);
TH_API void THTensor_(randperm)(THTensor *r_, long n);

TH_API void THTensor_(reshape)(THTensor *r_, THTensor *t, THLongStorage *size);
TH_API void THTensor_(sort)(THTensor *rt_, THLongTensor *ri_, THTensor *t, int dimension, int descendingOrder);
TH_API void THTensor_(tril)(THTensor *r_, THTensor *t, long k);
TH_API void THTensor_(triu)(THTensor *r_, THTensor *t, long k);
TH_API void THTensor_(cat)(THTensor *r_, THTensor *ta, THTensor *tb, int dimension);

TH_API void THTensor_(ltValue)(THByteTensor *r_, THTensor* t, real value);
TH_API void THTensor_(leValue)(THByteTensor *r_, THTensor* t, real value);
TH_API void THTensor_(gtValue)(THByteTensor *r_, THTensor* t, real value);
TH_API void THTensor_(geValue)(THByteTensor *r_, THTensor* t, real value);
TH_API void THTensor_(neValue)(THByteTensor *r_, THTensor* t, real value);
TH_API void THTensor_(eqValue)(THByteTensor *r_, THTensor* t, real value);

TH_API void THTensor_(ltTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
TH_API void THTensor_(leTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
TH_API void THTensor_(gtTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
TH_API void THTensor_(geTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
TH_API void THTensor_(neTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);
TH_API void THTensor_(eqTensor)(THByteTensor *r_, THTensor *ta, THTensor *tb);

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

TH_API void THTensor_(log)(THTensor *r_, THTensor *t);
TH_API void THTensor_(log1p)(THTensor *r_, THTensor *t);
TH_API void THTensor_(exp)(THTensor *r_, THTensor *t);
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
TH_API void THTensor_(pow)(THTensor *r_, THTensor *t, real value);
TH_API void THTensor_(sqrt)(THTensor *r_, THTensor *t);
TH_API void THTensor_(ceil)(THTensor *r_, THTensor *t);
TH_API void THTensor_(floor)(THTensor *r_, THTensor *t);
TH_API void THTensor_(abs)(THTensor *r_, THTensor *t);

TH_API void THTensor_(mean)(THTensor *r_, THTensor *t, int dimension);
TH_API void THTensor_(std)(THTensor *r_, THTensor *t, int dimension, int flag);
TH_API void THTensor_(var)(THTensor *r_, THTensor *t, int dimension, int flag);
TH_API void THTensor_(norm)(THTensor *r_, THTensor *t, real value, int dimension);
TH_API accreal THTensor_(dist)(THTensor *a, THTensor *b, real value);
TH_API void THTensor_(histc)(THTensor *hist, THTensor *tensor, long nbins, real minvalue, real maxvalue);

TH_API accreal THTensor_(meanall)(THTensor *self);
TH_API accreal THTensor_(varall)(THTensor *self);
TH_API accreal THTensor_(stdall)(THTensor *self);
TH_API accreal THTensor_(normall)(THTensor *t, real value);

TH_API void THTensor_(linspace)(THTensor *r_, real a, real b, long n);
TH_API void THTensor_(logspace)(THTensor *r_, real a, real b, long n);
TH_API void THTensor_(rand)(THTensor *r_, THLongStorage *size);
TH_API void THTensor_(randn)(THTensor *r_, THLongStorage *size);

#endif

#endif

#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZTensorMath.h"
#else

TH_API void THZTensor_(fill)(THZTensor *r_, ntype value);
TH_API void THZTensor_(zero)(THZTensor *r_);

TH_API void THZTensor_(maskedFill)(THZTensor *tensor, THByteTensor *mask, ntype value);
TH_API void THZTensor_(maskedCopy)(THZTensor *tensor, THByteTensor *mask, THZTensor* src);
TH_API void THZTensor_(maskedSelect)(THZTensor *tensor, THZTensor* src, THByteTensor *mask);

TH_API void THZTensor_(nonzero)(THLongTensor *subscript, THZTensor *tensor);

TH_API void THZTensor_(indexSelect)(THZTensor *tensor, THZTensor *src, int dim, THLongTensor *index);
TH_API void THZTensor_(indexCopy)(THZTensor *tensor, int dim, THLongTensor *index, THZTensor *src);
TH_API void THZTensor_(indexAdd)(THZTensor *tensor, int dim, THLongTensor *index, THZTensor *src);
TH_API void THZTensor_(indexFill)(THZTensor *tensor, int dim, THLongTensor *index, ntype val);
TH_API void THZTensor_(take)(THZTensor *tensor, THZTensor *src, THLongTensor *index);
TH_API void THZTensor_(put)(THZTensor *tensor, THLongTensor *index, THZTensor *src, int accumulate);

TH_API void THZTensor_(gather)(THZTensor *tensor, THZTensor *src, int dim, THLongTensor *index);
TH_API void THZTensor_(scatter)(THZTensor *tensor, int dim, THLongTensor *index, THZTensor *src);
TH_API void THZTensor_(scatterAdd)(THZTensor *tensor, int dim, THLongTensor *index, THZTensor *src);
TH_API void THZTensor_(scatterFill)(THZTensor *tensor, int dim, THLongTensor *index, ntype val);

TH_API accntype THZTensor_(dot)(THZTensor *t, THZTensor *src);

TH_API ntype THZTensor_(minall)(THZTensor *t);
TH_API ntype THZTensor_(maxall)(THZTensor *t);
TH_API ntype THZTensor_(medianall)(THZTensor *t);
TH_API accntype THZTensor_(sumall)(THZTensor *t);
TH_API accntype THZTensor_(prodall)(THZTensor *t);

TH_API void THZTensor_(neg)(THZTensor *self, THZTensor *src);
TH_API void THZTensor_(cinv)(THZTensor *self, THZTensor *src);

TH_API void THZTensor_(add)(THZTensor *r_, THZTensor *t, ntype value);
TH_API void THZTensor_(sub)(THZTensor *r_, THZTensor *t, ntype value);
TH_API void THZTensor_(add_scaled)(THZTensor *r_, THZTensor *t, ntype value, ntype alpha);
TH_API void THZTensor_(sub_scaled)(THZTensor *r_, THZTensor *t, ntype value, ntype alpha);
TH_API void THZTensor_(mul)(THZTensor *r_, THZTensor *t, ntype value);
TH_API void THZTensor_(div)(THZTensor *r_, THZTensor *t, ntype value);
TH_API void THZTensor_(lshift)(THZTensor *r_, THZTensor *t, ntype value);
TH_API void THZTensor_(rshift)(THZTensor *r_, THZTensor *t, ntype value);
TH_API void THZTensor_(fmod)(THZTensor *r_, THZTensor *t, ntype value);
TH_API void THZTensor_(remainder)(THZTensor *r_, THZTensor *t, ntype value);
TH_API void THZTensor_(clamp)(THZTensor *r_, THZTensor *t, ntype min_value, ntype max_value);
TH_API void THZTensor_(bitand)(THZTensor *r_, THZTensor *t, ntype value);
TH_API void THZTensor_(bitor)(THZTensor *r_, THZTensor *t, ntype value);
TH_API void THZTensor_(bitxor)(THZTensor *r_, THZTensor *t, ntype value);

TH_API void THZTensor_(cadd)(THZTensor *r_, THZTensor *t, ntype value, THZTensor *src);
TH_API void THZTensor_(csub)(THZTensor *self, THZTensor *src1, ntype value, THZTensor *src2);
TH_API void THZTensor_(cmul)(THZTensor *r_, THZTensor *t, THZTensor *src);
TH_API void THZTensor_(cpow)(THZTensor *r_, THZTensor *t, THZTensor *src);
TH_API void THZTensor_(cdiv)(THZTensor *r_, THZTensor *t, THZTensor *src);
TH_API void THZTensor_(clshift)(THZTensor *r_, THZTensor *t, THZTensor *src);
TH_API void THZTensor_(crshift)(THZTensor *r_, THZTensor *t, THZTensor *src);
TH_API void THZTensor_(cfmod)(THZTensor *r_, THZTensor *t, THZTensor *src);
TH_API void THZTensor_(cremainder)(THZTensor *r_, THZTensor *t, THZTensor *src);
TH_API void THZTensor_(cbitand)(THZTensor *r_, THZTensor *t, THZTensor *src);
TH_API void THZTensor_(cbitor)(THZTensor *r_, THZTensor *t, THZTensor *src);
TH_API void THZTensor_(cbitxor)(THZTensor *r_, THZTensor *t, THZTensor *src);

TH_API void THZTensor_(addcmul)(THZTensor *r_, THZTensor *t, ntype value, THZTensor *src1, THZTensor *src2);
TH_API void THZTensor_(addcdiv)(THZTensor *r_, THZTensor *t, ntype value, THZTensor *src1, THZTensor *src2);

TH_API void THZTensor_(addmv)(THZTensor *r_, ntype beta, THZTensor *t, ntype alpha, THZTensor *mat,  THZTensor *vec);
TH_API void THZTensor_(addmm)(THZTensor *r_, ntype beta, THZTensor *t, ntype alpha, THZTensor *mat1, THZTensor *mat2);
TH_API void THZTensor_(addr)(THZTensor *r_,  ntype beta, THZTensor *t, ntype alpha, THZTensor *vec1, THZTensor *vec2);

TH_API void THZTensor_(addbmm)(THZTensor *r_, ntype beta, THZTensor *t, ntype alpha, THZTensor *batch1, THZTensor *batch2);
TH_API void THZTensor_(baddbmm)(THZTensor *r_, ntype beta, THZTensor *t, ntype alpha, THZTensor *batch1, THZTensor *batch2);

TH_API void THZTensor_(match)(THZTensor *r_, THZTensor *m1, THZTensor *m2, ntype gain);

TH_API ptrdiff_t THZTensor_(numel)(THZTensor *t);
TH_API void THZTensor_(max)(THZTensor *values_, THLongTensor *indices_, THZTensor *t, int dimension, int keepdim);
TH_API void THZTensor_(min)(THZTensor *values_, THLongTensor *indices_, THZTensor *t, int dimension, int keepdim);
TH_API void THZTensor_(kthvalue)(THZTensor *values_, THLongTensor *indices_, THZTensor *t, int64_t k, int dimension, int keepdim);
TH_API void THZTensor_(mode)(THZTensor *values_, THLongTensor *indices_, THZTensor *t, int dimension, int keepdim);
TH_API void THZTensor_(median)(THZTensor *values_, THLongTensor *indices_, THZTensor *t, int dimension, int keepdim);
TH_API void THZTensor_(sum)(THZTensor *r_, THZTensor *t, int dimension, int keepdim);
TH_API void THZTensor_(prod)(THZTensor *r_, THZTensor *t, int dimension, int keepdim);
TH_API void THZTensor_(cumsum)(THZTensor *r_, THZTensor *t, int dimension);
TH_API void THZTensor_(cumprod)(THZTensor *r_, THZTensor *t, int dimension);
TH_API void THZTensor_(sign)(THZTensor *r_, THZTensor *t);
TH_API accntype THZTensor_(trace)(THZTensor *t);
TH_API void THZTensor_(cross)(THZTensor *r_, THZTensor *a, THZTensor *b, int dimension);

TH_API void THZTensor_(cmax)(THZTensor *r, THZTensor *t, THZTensor *src);
TH_API void THZTensor_(cmin)(THZTensor *r, THZTensor *t, THZTensor *src);
TH_API void THZTensor_(cmaxValue)(THZTensor *r, THZTensor *t, ntype value);
TH_API void THZTensor_(cminValue)(THZTensor *r, THZTensor *t, ntype value);

TH_API void THZTensor_(zeros)(THZTensor *r_, THLongStorage *size);
TH_API void THZTensor_(zerosLike)(THZTensor *r_, THZTensor *input);
TH_API void THZTensor_(ones)(THZTensor *r_, THLongStorage *size);
TH_API void THZTensor_(onesLike)(THZTensor *r_, THZTensor *input);
TH_API void THZTensor_(diag)(THZTensor *r_, THZTensor *t, int k);
TH_API void THZTensor_(eye)(THZTensor *r_, int64_t n, int64_t m);
TH_API void THZTensor_(arange)(THZTensor *r_, accntype xmin, accntype xmax, accntype step);
TH_API void THZTensor_(range)(THZTensor *r_, accntype xmin, accntype xmax, accntype step);
TH_API void THZTensor_(randperm)(THZTensor *r_, THGenerator *_generator, int64_t n);

TH_API void THZTensor_(reshape)(THZTensor *r_, THZTensor *t, THLongStorage *size);
TH_API void THZTensor_(sort)(THZTensor *rt_, THLongTensor *ri_, THZTensor *t, int dimension, int descendingOrder);
TH_API void THZTensor_(topk)(THZTensor *rt_, THLongTensor *ri_, THZTensor *t, int64_t k, int dim, int dir, int sorted);
TH_API void THZTensor_(tril)(THZTensor *r_, THZTensor *t, int64_t k);
TH_API void THZTensor_(triu)(THZTensor *r_, THZTensor *t, int64_t k);
TH_API void THZTensor_(cat)(THZTensor *r_, THZTensor *ta, THZTensor *tb, int dimension);
TH_API void THZTensor_(catArray)(THZTensor *result, THZTensor **inputs, int numInputs, int dimension);

TH_API int THZTensor_(equal)(THZTensor *ta, THZTensor *tb);

TH_API void THZTensor_(ltValue)(THByteTensor *r_, THZTensor* t, ntype value);
TH_API void THZTensor_(leValue)(THByteTensor *r_, THZTensor* t, ntype value);
TH_API void THZTensor_(gtValue)(THByteTensor *r_, THZTensor* t, ntype value);
TH_API void THZTensor_(geValue)(THByteTensor *r_, THZTensor* t, ntype value);
TH_API void THZTensor_(neValue)(THByteTensor *r_, THZTensor* t, ntype value);
TH_API void THZTensor_(eqValue)(THByteTensor *r_, THZTensor* t, ntype value);

TH_API void THZTensor_(ltValueT)(THZTensor *r_, THZTensor* t, ntype value);
TH_API void THZTensor_(leValueT)(THZTensor *r_, THZTensor* t, ntype value);
TH_API void THZTensor_(gtValueT)(THZTensor *r_, THZTensor* t, ntype value);
TH_API void THZTensor_(geValueT)(THZTensor *r_, THZTensor* t, ntype value);
TH_API void THZTensor_(neValueT)(THZTensor *r_, THZTensor* t, ntype value);
TH_API void THZTensor_(eqValueT)(THZTensor *r_, THZTensor* t, ntype value);

TH_API void THZTensor_(ltTensor)(THByteTensor *r_, THZTensor *ta, THZTensor *tb);
TH_API void THZTensor_(leTensor)(THByteTensor *r_, THZTensor *ta, THZTensor *tb);
TH_API void THZTensor_(gtTensor)(THByteTensor *r_, THZTensor *ta, THZTensor *tb);
TH_API void THZTensor_(geTensor)(THByteTensor *r_, THZTensor *ta, THZTensor *tb);
TH_API void THZTensor_(neTensor)(THByteTensor *r_, THZTensor *ta, THZTensor *tb);
TH_API void THZTensor_(eqTensor)(THByteTensor *r_, THZTensor *ta, THZTensor *tb);

TH_API void THZTensor_(ltTensorT)(THZTensor *r_, THZTensor *ta, THZTensor *tb);
TH_API void THZTensor_(leTensorT)(THZTensor *r_, THZTensor *ta, THZTensor *tb);
TH_API void THZTensor_(gtTensorT)(THZTensor *r_, THZTensor *ta, THZTensor *tb);
TH_API void THZTensor_(geTensorT)(THZTensor *r_, THZTensor *ta, THZTensor *tb);
TH_API void THZTensor_(neTensorT)(THZTensor *r_, THZTensor *ta, THZTensor *tb);
TH_API void THZTensor_(eqTensorT)(THZTensor *r_, THZTensor *ta, THZTensor *tb);

TH_API void THZTensor_(sigmoid)(THZTensor *r_, THZTensor *t);
TH_API void THZTensor_(log)(THZTensor *r_, THZTensor *t);
TH_API void THZTensor_(log1p)(THZTensor *r_, THZTensor *t);
TH_API void THZTensor_(exp)(THZTensor *r_, THZTensor *t);
TH_API void THZTensor_(cos)(THZTensor *r_, THZTensor *t);
TH_API void THZTensor_(acos)(THZTensor *r_, THZTensor *t);
TH_API void THZTensor_(cosh)(THZTensor *r_, THZTensor *t);
TH_API void THZTensor_(sin)(THZTensor *r_, THZTensor *t);
TH_API void THZTensor_(asin)(THZTensor *r_, THZTensor *t);
TH_API void THZTensor_(sinh)(THZTensor *r_, THZTensor *t);
TH_API void THZTensor_(tan)(THZTensor *r_, THZTensor *t);
TH_API void THZTensor_(atan)(THZTensor *r_, THZTensor *t);
TH_API void THZTensor_(atan2)(THZTensor *r_, THZTensor *tx, THZTensor *ty);
TH_API void THZTensor_(tanh)(THZTensor *r_, THZTensor *t);
TH_API void THZTensor_(pow)(THZTensor *r_, THZTensor *t, ntype value);
TH_API void THZTensor_(tpow)(THZTensor *r_, ntype value, THZTensor *t);
TH_API void THZTensor_(sqrt)(THZTensor *r_, THZTensor *t);
TH_API void THZTensor_(rsqrt)(THZTensor *r_, THZTensor *t);
TH_API void THZTensor_(lerp)(THZTensor *r_, THZTensor *a, THZTensor *b, ntype weight);

TH_API void THZTensor_(mean)(THZTensor *r_, THZTensor *t, int dimension, int keepdim);
TH_API void THZTensor_(std)(THZTensor *r_, THZTensor *t, int dimension, int biased, int keepdim);
TH_API void THZTensor_(var)(THZTensor *r_, THZTensor *t, int dimension, int biased, int keepdim);
TH_API void THZTensor_(norm)(THZTensor *r_, THZTensor *t, ntype value, int dimension, int keepdim);
TH_API void THZTensor_(renorm)(THZTensor *r_, THZTensor *t, ntype value, int dimension, ntype maxnorm);
TH_API accntype THZTensor_(dist)(THZTensor *a, THZTensor *b, ntype value);
TH_API void THZTensor_(histc)(THZTensor *hist, THZTensor *tensor, int64_t nbins, ntype minvalue, ntype maxvalue);
TH_API void THZTensor_(bhistc)(THZTensor *hist, THZTensor *tensor, int64_t nbins, ntype minvalue, ntype maxvalue);

TH_API accntype THZTensor_(meanall)(THZTensor *self);
TH_API accntype THZTensor_(varall)(THZTensor *self, int biased);
TH_API accntype THZTensor_(stdall)(THZTensor *self, int biased);
TH_API accntype THZTensor_(normall)(THZTensor *t, ntype value);

TH_API void THZTensor_(linspace)(THZTensor *r_, ntype a, ntype b, int64_t n);
TH_API void THZTensor_(logspace)(THZTensor *r_, ntype a, ntype b, int64_t n);
TH_API void THZTensor_(rand)(THZTensor *r_, THGenerator *_generator, THLongStorage *size);
TH_API void THZTensor_(randn)(THZTensor *r_, THGenerator *_generator, THLongStorage *size);

TH_API void THZTensor_(abs)(THZPartTensor *r_, THZTensor *t);
TH_API void THZTensor_(real)(THZPartTensor *r_, THZTensor *t);
TH_API void THZTensor_(imag)(THZPartTensor *r_, THZTensor *t);
TH_API void THZTensor_(arg)(THZPartTensor *r_, THZTensor *t);
TH_API void THZTensor_(proj)(THZPartTensor *r_, THZTensor *t);

TH_API void THZTensor_(conj)(THZTensor *r_, THZTensor *t);

#endif

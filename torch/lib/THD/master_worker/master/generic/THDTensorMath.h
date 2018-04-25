#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDTensorMath.h"
#else

THD_API void THDTensor_(gather)(THDTensor *self, THDTensor *src, int dim,
                                THDLongTensor *index);
THD_API void THDTensor_(scatter)(THDTensor *self, int dim, THDLongTensor *index,
                                 THDTensor *src);
THD_API void THDTensor_(scatterFill)(THDTensor *self, int dim,
                                     THDLongTensor *index, real val);
THD_API void THDTensor_(scatterAdd)(THDTensor *self, int dim, THDLongTensor *index,
                                 THDTensor *src);

THD_API void THDTensor_(max)(THDTensor *self, THDLongTensor *indices_,
                             THDTensor *src, int dimension, int keepdim);
THD_API void THDTensor_(min)(THDTensor *self, THDLongTensor *indices_,
                             THDTensor *src, int dimension, int keepdim);
THD_API void THDTensor_(kthvalue)(THDTensor *self, THDLongTensor *indices_,
                                  THDTensor *src, int64_t k, int dimension, int keepdim);
THD_API void THDTensor_(mode)(THDTensor *self, THDLongTensor *indices_,
                              THDTensor *src, int dimension, int keepdim);
THD_API void THDTensor_(median)(THDTensor *self, THDLongTensor *indices_,
                                THDTensor *src, int dimension, int keepdim);

THD_API void THDTensor_(fill)(THDTensor *r_, real value);
THD_API void THDTensor_(zero)(THDTensor *r);
THD_API void THDTensor_(maskedFill)(THDTensor *tensor, THDByteTensor *mask, real value);
THD_API void THDTensor_(maskedCopy)(THDTensor *tensor, THDByteTensor *mask, THDTensor* src);
THD_API void THDTensor_(maskedSelect)(THDTensor *tensor, THDTensor* src, THDByteTensor *mask);
THD_API void THDTensor_(nonzero)(THDLongTensor *subscript, THDTensor *tensor);
THD_API void THDTensor_(indexSelect)(THDTensor *tensor, THDTensor *src, int dim,
                                     THDLongTensor *index);
THD_API void THDTensor_(indexCopy)(THDTensor *tensor, int dim, THDLongTensor *index, THDTensor *src);
THD_API void THDTensor_(indexAdd)(THDTensor *tensor, int dim, THDLongTensor *index, THDTensor *src);
THD_API void THDTensor_(indexFill)(THDTensor *tensor, int dim, THDLongTensor *index, real val);


THD_API void THDTensor_(zeros)(THDTensor *r_, THLongStorage *size);
THD_API void THDTensor_(ones)(THDTensor *r_, THLongStorage *size);
THD_API ptrdiff_t THDTensor_(numel)(THDTensor *t);

THD_API void THDTensor_(diag)(THDTensor *r_, THDTensor *t, int k);
THD_API void THDTensor_(eye)(THDTensor *r_, int64_t n, int64_t m);
THD_API void THDTensor_(range)(THDTensor *r_, accreal xmin, accreal xmax, accreal step);
THD_API void THDTensor_(randperm)(THDTensor *r_, THDGenerator *_generator, int64_t n);
THD_API void THDTensor_(reshape)(THDTensor *r_, THDTensor *t, THDLongStorage *size);
THD_API void THDTensor_(sort)(THDTensor *rt_, THDLongTensor *ri_,
                              THDTensor *t, int dimension,
                              int descendingOrder);
THD_API void THDTensor_(topk)(THDTensor *rt_, THDLongTensor *ri_,
                              THDTensor *t, int64_t k, int dim,
                              int dir, int sorted);
THD_API void THDTensor_(tril)(THDTensor *r_, THDTensor *t, int64_t k);
THD_API void THDTensor_(triu)(THDTensor *r_, THDTensor *t, int64_t k);
THD_API void THDTensor_(cat)(THDTensor *r_, THDTensor *ta,
                             THDTensor *tb, int dimension);
THD_API void THDTensor_(catArray)(THDTensor *result, THDTensor **inputs,
                                  int numInputs, int dimension);
THD_API int THDTensor_(equal)(THDTensor *ta, THDTensor *tb);
THD_API void THDTensor_(ltValue)(THDByteTensor *r_, THDTensor *t, real value);
THD_API void THDTensor_(leValue)(THDByteTensor *r_, THDTensor *t, real value);
THD_API void THDTensor_(gtValue)(THDByteTensor *r_, THDTensor *t, real value);
THD_API void THDTensor_(geValue)(THDByteTensor *r_, THDTensor *t, real value);
THD_API void THDTensor_(neValue)(THDByteTensor *r_, THDTensor *t, real value);
THD_API void THDTensor_(eqValue)(THDByteTensor *r_, THDTensor *t, real value);
THD_API void THDTensor_(ltValueT)(THDTensor *r_, THDTensor *t, real value);
THD_API void THDTensor_(leValueT)(THDTensor *r_, THDTensor *t, real value);
THD_API void THDTensor_(gtValueT)(THDTensor *r_, THDTensor *t, real value);
THD_API void THDTensor_(geValueT)(THDTensor *r_, THDTensor *t, real value);
THD_API void THDTensor_(neValueT)(THDTensor *r_, THDTensor *t, real value);
THD_API void THDTensor_(eqValueT)(THDTensor *r_, THDTensor *t, real value);
THD_API void THDTensor_(ltTensor)(THDByteTensor *r_, THDTensor *ta, THDTensor *tb);
THD_API void THDTensor_(leTensor)(THDByteTensor *r_, THDTensor *ta, THDTensor *tb);
THD_API void THDTensor_(gtTensor)(THDByteTensor *r_, THDTensor *ta, THDTensor *tb);
THD_API void THDTensor_(geTensor)(THDByteTensor *r_, THDTensor *ta, THDTensor *tb);
THD_API void THDTensor_(neTensor)(THDByteTensor *r_, THDTensor *ta, THDTensor *tb);
THD_API void THDTensor_(eqTensor)(THDByteTensor *r_, THDTensor *ta, THDTensor *tb);
THD_API void THDTensor_(ltTensorT)(THDTensor *r_, THDTensor *ta, THDTensor *tb);
THD_API void THDTensor_(leTensorT)(THDTensor *r_, THDTensor *ta, THDTensor *tb);
THD_API void THDTensor_(gtTensorT)(THDTensor *r_, THDTensor *ta, THDTensor *tb);
THD_API void THDTensor_(geTensorT)(THDTensor *r_, THDTensor *ta, THDTensor *tb);
THD_API void THDTensor_(neTensorT)(THDTensor *r_, THDTensor *ta, THDTensor *tb);
THD_API void THDTensor_(eqTensorT)(THDTensor *r_, THDTensor *ta, THDTensor *tb);
THD_API void THDTensor_(abs)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(sigmoid)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(log)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(log10)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(log1p)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(log2)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(exp)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(expm1)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(cos)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(acos)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(cosh)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(sin)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(asin)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(sinh)(THDTensor *r_, THDTensor *t);

THD_API void THDTensor_(tan)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(atan)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(atan2)(THDTensor *r_, THDTensor *tx, THDTensor *ty);
THD_API void THDTensor_(tanh)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(pow)(THDTensor *r_, THDTensor *t, real value);
THD_API void THDTensor_(tpow)(THDTensor *r_, real value, THDTensor *t);
THD_API void THDTensor_(sqrt)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(rsqrt)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(ceil)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(floor)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(round)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(abs)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(trunc)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(frac)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(lerp)(THDTensor *r_, THDTensor *a, THDTensor *b, real weight);
THD_API void THDTensor_(mean)(THDTensor *r_, THDTensor *t, int dimension, int keepdim);
THD_API void THDTensor_(std)(THDTensor *r_, THDTensor *t, int dimension, int biased, int keepdim);
THD_API void THDTensor_(var)(THDTensor *r_, THDTensor *t, int dimension, int biased, int keepdim);
THD_API void THDTensor_(norm)(THDTensor *r_, THDTensor *t, real value,
                              int dimension, int keepdim);
THD_API void THDTensor_(renorm)(THDTensor *r_, THDTensor *t, real value,
                                int dimension, real maxnorm);
THD_API accreal THDTensor_(dist)(THDTensor *a, THDTensor *b, real value);
THD_API void THDTensor_(histc)(THDTensor *hist, THDTensor *tensor, int64_t nbins,
                               real minvalue, real maxvalue);
THD_API void THDTensor_(bhistc)(THDTensor *hist, THDTensor *tensor, int64_t nbins,
                                real minvalue, real maxvalue);
THD_API accreal THDTensor_(meanall)(THDTensor *self);
THD_API accreal THDTensor_(varall)(THDTensor *self, int biased);
THD_API accreal THDTensor_(stdall)(THDTensor *self, int biased);
THD_API accreal THDTensor_(normall)(THDTensor *t, real value);
THD_API void THDTensor_(linspace)(THDTensor *r_, real a, real b, int64_t n);
THD_API void THDTensor_(logspace)(THDTensor *r_, real a, real b, int64_t n);
THD_API void THDTensor_(rand)(THDTensor *r_, THDGenerator *_generator,
                              THLongStorage *size);
THD_API void THDTensor_(randn)(THDTensor *r_, THDGenerator *_generator,
                               THLongStorage *size);
THD_API int THDTensor_(logicalAll)(THDTensor *self);
THD_API int THDTensor_(logicalAny)(THDTensor *self);

THD_API void THDTensor_(clshift)(THDTensor *r_, THDTensor *t, THDTensor *src);
THD_API void THDTensor_(crshift)(THDTensor *r_, THDTensor *t, THDTensor *src);
THD_API void THDTensor_(cbitand)(THDTensor *r_, THDTensor *t, THDTensor *src);
THD_API void THDTensor_(cbitor)(THDTensor *r_, THDTensor *t, THDTensor *src);
THD_API void THDTensor_(cbitxor)(THDTensor *r_, THDTensor *t, THDTensor *src);
THD_API void THDTensor_(lshift)(THDTensor *r_, THDTensor *t, real value);
THD_API void THDTensor_(rshift)(THDTensor *r_, THDTensor *t, real value);
THD_API void THDTensor_(bitand)(THDTensor *r_, THDTensor *t, real value);
THD_API void THDTensor_(bitor)(THDTensor *r_, THDTensor *t, real value);
THD_API void THDTensor_(bitxor)(THDTensor *r_, THDTensor *t, real value);

#endif

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDTensorMath.h"
#else

THD_API void THDTensor_(fill)(THDTensor *r_, real value);
THD_API void THDTensor_(zeros)(THDTensor *r_, THLongStorage *size);
THD_API void THDTensor_(ones)(THDTensor *r_, THLongStorage *size);
THD_API ptrdiff_t THDTensor_(numel)(THDTensor *t);

THD_API void THDTensor_(diag)(THDTensor *r_, THDTensor *t, int k);
THD_API void THDTensor_(eye)(THDTensor *r_, long n, long m);
THD_API void THDTensor_(range)(THDTensor *r_, accreal xmin, accreal xmax, accreal step);
// TODO figure out how to implement randomized functions
//THD_API void THDTensor_(randperm)(THDTensor *r_, THDGenerator *_generator, long n);
THD_API void THDTensor_(reshape)(THDTensor *r_, THDTensor *t, THDLongStorage *size);
THD_API void THDTensor_(sort)(THDTensor *rt_, THDLongTensor *ri_,
                              THDTensor *t, int dimension,
                              int descendingOrder);
THD_API void THDTensor_(topk)(THDTensor *rt_, THDLongTensor *ri_,
                              THDTensor *t, long k, int dim,
                              int dir, int sorted);
THD_API void THDTensor_(tril)(THDTensor *r_, THDTensor *t, long k);
THD_API void THDTensor_(triu)(THDTensor *r_, THDTensor *t, long k);
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
THD_API void THDTensor_(log1p)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(exp)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(cos)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(acos)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(cosh)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(sin)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(asin)(THDTensor *r_, THDTensor *t);
THD_API void THDTensor_(sinh)(THDTensor *r_, THDTensor *t);

#endif

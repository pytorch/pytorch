#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorRandom.h"
#else

#include <ATen/core/Generator.h>
#include <ATen/core/DistributionsHelper.h>

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_BFLOAT16)
TH_API void THTensor_(uniform)(THTensor *self, double a, double b, at::Generator *_generator);
#if !defined(TH_REAL_IS_BFLOAT16)
TH_API void THTensor_(multinomialAliasSetup)(THTensor *prob_dist, THLongTensor *J, THTensor *q);
TH_API void THTensor_(multinomialAliasDraw)(THLongTensor *self, THTensor *q, THLongTensor *J, int n_sample, at::Generator *_generator);
#endif
#endif

#if defined(TH_REAL_IS_BYTE)
TH_API void THTensor_(getRNGState)(at::Generator *_generator, THTensor *self);
TH_API void THTensor_(setRNGState)(at::Generator *_generator, THTensor *self);
#endif

#endif

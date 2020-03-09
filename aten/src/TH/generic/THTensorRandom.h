#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorRandom.h"
#else

#include <ATen/core/Generator.h>
#include <ATen/core/DistributionsHelper.h>

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
TH_API void THTensor_(uniform)(THTensor *self, double a, double b, at::GeneratorHolder _generator);
TH_API void THTensor_(multinomialAliasSetup)(THTensor *prob_dist, THLongTensor *J, THTensor *q);
TH_API void THTensor_(multinomialAliasDraw)(THLongTensor *self, THTensor *q, THLongTensor *J, int n_sample, at::GeneratorHolder _generator);
#endif

#if defined(TH_REAL_IS_BYTE)
TH_API void THTensor_(getRNGState)(at::GeneratorHolder _generator, THTensor *self);
TH_API void THTensor_(setRNGState)(at::GeneratorHolder _generator, THTensor *self);
#endif

#endif

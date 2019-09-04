#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorRandom.h"
#else

#include <ATen/core/Generator.h>
#include <ATen/core/DistributionsHelper.h>

TH_API void THTensor_(random)(THTensor *self, at::Generator *_generator);
TH_API void THTensor_(clampedRandom)(THTensor *self, int64_t min, int64_t max, at::Generator *_generator);
TH_API void THTensor_(cappedRandom)(THTensor *self, int64_t max, at::Generator *_generator);
TH_API void THTensor_(geometric)(THTensor *self, double p, at::Generator *_generator);

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
TH_API void THTensor_(bernoulli_Tensor)(THTensor *self, at::Generator *_generator, THTensor *p);
TH_API void THTensor_(uniform)(THTensor *self, double a, double b, at::Generator *_generator);
TH_API void THTensor_(normal)(THTensor *self, double mean, double stdv, at::Generator *_generator);
TH_API void THTensor_(normal_means)(THTensor *self, THTensor *means, double stddev, at::Generator *gen);
TH_API void THTensor_(normal_stddevs)(THTensor *self, double mean, THTensor *stddevs, at::Generator *gen);
TH_API void THTensor_(normal_means_stddevs)(THTensor *self, THTensor *means, THTensor *stddevs, at::Generator *gen);
TH_API void THTensor_(exponential)(THTensor *self, double lambda, at::Generator *_generator);
TH_API void THTensor_(cauchy)(THTensor *self, double median, double sigma, at::Generator *_generator);
TH_API void THTensor_(logNormal)(THTensor *self, double mean, double stdv, at::Generator *_generator);
TH_API void THTensor_(multinomialAliasSetup)(THTensor *prob_dist, THLongTensor *J, THTensor *q);
TH_API void THTensor_(multinomialAliasDraw)(THLongTensor *self, THTensor *q, THLongTensor *J, int n_sample, at::Generator *_generator);
#endif

#if defined(TH_REAL_IS_BYTE)
TH_API void THTensor_(getRNGState)(at::Generator *_generator, THTensor *self);
TH_API void THTensor_(setRNGState)(at::Generator *_generator, THTensor *self);
#endif

#endif

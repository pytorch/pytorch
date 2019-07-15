#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorRandom.h"
#else

#include <ATen/core/Generator.h>
#include <ATen/core/DistributionsHelper.h>

TH_API void THTensor_(random)(THTensor *self, at::Generator *_generator);
TH_API void THTensor_(clampedRandom)(THTensor *self, at::Generator *_generator, int64_t min, int64_t max);
TH_API void THTensor_(cappedRandom)(THTensor *self, at::Generator *_generator, int64_t max);
TH_API void THTensor_(geometric)(THTensor *self, at::Generator *_generator, double p);

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
TH_API void THTensor_(bernoulli_Tensor)(THTensor *self, at::Generator *_generator, THTensor *p);
TH_API void THTensor_(uniform)(THTensor *self, at::Generator *_generator, double a, double b);
TH_API void THTensor_(normal)(THTensor *self, at::Generator *_generator, double mean, double stdv);
TH_API void THTensor_(normal_means)(THTensor *self, at::Generator *gen, THTensor *means, double stddev);
TH_API void THTensor_(normal_stddevs)(THTensor *self, at::Generator *gen, double mean, THTensor *stddevs);
TH_API void THTensor_(normal_means_stddevs)(THTensor *self, at::Generator *gen, THTensor *means, THTensor *stddevs);
TH_API void THTensor_(exponential)(THTensor *self, at::Generator *_generator, double lambda);
TH_API void THTensor_(cauchy)(THTensor *self, at::Generator *_generator, double median, double sigma);
TH_API void THTensor_(logNormal)(THTensor *self, at::Generator *_generator, double mean, double stdv);
TH_API void THTensor_(multinomial)(THLongTensor *self, at::Generator *_generator, THTensor *prob_dist, int n_sample, int with_replacement);
TH_API void THTensor_(multinomialAliasSetup)(THTensor *prob_dist, THLongTensor *J, THTensor *q);
TH_API void THTensor_(multinomialAliasDraw)(THLongTensor *self, at::Generator *_generator, THTensor *q, THLongTensor *J, int n_sample);
#endif

#if defined(TH_REAL_IS_BYTE)
TH_API void THTensor_(getRNGState)(at::Generator *_generator, THTensor *self);
TH_API void THTensor_(setRNGState)(at::Generator *_generator, THTensor *self);
#endif

#endif

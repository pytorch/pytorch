#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorRandom.h"
#else

TH_API void THTensor_(random)(THTensor *self);
TH_API void THTensor_(geometric)(THTensor *self, double p);
TH_API void THTensor_(bernoulli)(THTensor *self, double p);

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
TH_API void THTensor_(uniform)(THTensor *self, double a, double b);
TH_API void THTensor_(normal)(THTensor *self, double mean, double stdv);
TH_API void THTensor_(exponential)(THTensor *self, double lambda);
TH_API void THTensor_(cauchy)(THTensor *self, double median, double sigma);
TH_API void THTensor_(logNormal)(THTensor *self, double mean, double stdv);
#endif

#endif

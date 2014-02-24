#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorRandom.h"
#else

TH_API void THTensor_(random)(mersenne_state *_mersenne, THTensor *self);
TH_API void THTensor_(geometric)(mersenne_state *_mersenne, THTensor *self, double p);
TH_API void THTensor_(bernoulli)(mersenne_state *_mersenne, THTensor *self, double p);

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
TH_API void THTensor_(uniform)(mersenne_state *_mersenne, THTensor *self, double a, double b);
TH_API void THTensor_(normal)(mersenne_state *_mersenne, THTensor *self, double mean, double stdv);
TH_API void THTensor_(exponential)(mersenne_state *_mersenne, THTensor *self, double lambda);
TH_API void THTensor_(cauchy)(mersenne_state *_mersenne, THTensor *self, double median, double sigma);
TH_API void THTensor_(logNormal)(mersenne_state *_mersenne, THTensor *self, double mean, double stdv);
#endif

#if defined(TH_REAL_IS_LONG)
TH_API void THTensor_(getRNGState)(mersenne_state *_mersenne, THTensor *self);
TH_API void THTensor_(setRNGState)(mersenne_state *_mersenne, THTensor *self);
#endif

#endif

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorRandom.h"
#else

TH_API void THTensor_(random)(THGenerator *_generator, THTensor *self);
TH_API void THTensor_(geometric)(THGenerator *_generator, THTensor *self, double p);
TH_API void THTensor_(bernoulli)(THGenerator *_generator, THTensor *self, double p);

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
TH_API void THTensor_(uniform)(THGenerator *_generator, THTensor *self, double a, double b);
TH_API void THTensor_(normal)(THGenerator *_generator, THTensor *self, double mean, double stdv);
TH_API void THTensor_(exponential)(THGenerator *_generator, THTensor *self, double lambda);
TH_API void THTensor_(cauchy)(THGenerator *_generator, THTensor *self, double median, double sigma);
TH_API void THTensor_(logNormal)(THGenerator *_generator, THTensor *self, double mean, double stdv);
#endif

#if defined(TH_REAL_IS_LONG)
TH_API void THTensor_(getRNGState)(THGenerator *_generator, THTensor *self);
TH_API void THTensor_(setRNGState)(THGenerator *_generator, THTensor *self);
#endif

#endif

#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZTensorRandom.h"
#else

TH_API void THZTensor_(random)(THZTensor *self, THGenerator *_generator);
TH_API void THZTensor_(uniform)(THZTensor *self, THGenerator *_generator, double a, double b);

#endif

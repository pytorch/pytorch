#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorLapack.h"
#else

TH_API void THTensor_(gels)(THTensor *rb_, THTensor *ra_, THTensor *b_, THTensor *a_);
TH_API void THTensor_(geev)(THTensor *re_, THTensor *rv_, THTensor *a_, bool eigenvectors);
TH_API void THTensor_(potri)(THTensor *ra_, THTensor *a, bool upper);
TH_API void THTensor_(geqrf)(THTensor *ra_, THTensor *rtau_, THTensor *a);
TH_API void THTensor_(orgqr)(THTensor *ra_, THTensor *a, THTensor *tau);
TH_API void THTensor_(ormqr)(THTensor *ra_, THTensor *a, THTensor *tau, THTensor *c, bool left, bool transpose);

#endif

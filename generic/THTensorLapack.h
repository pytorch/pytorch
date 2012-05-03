#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorLapack.h"
#else

TH_API void THTensor_(gesv)(THTensor *rb_, THTensor *ra_, THTensor *b_, THTensor *a_);
TH_API void THTensor_(gels)(THTensor *rb_, THTensor *ra_, THTensor *b_, THTensor *a_);
TH_API void THTensor_(syev)(THTensor *re_, THTensor *rv_, THTensor *a_, const char *jobz, const char *uplo);
TH_API void THTensor_(geev)(THTensor *re_, THTensor *rv_, THTensor *a_, const char *jobvr);
TH_API void THTensor_(gesvd)(THTensor *ru_, THTensor *rs_, THTensor *rv_, THTensor *a, const char *jobu);
TH_API void THTensor_(gesvd2)(THTensor *ru_, THTensor *rs_, THTensor *rv_, THTensor *ra_, THTensor *a, const char *jobu);
TH_API void THTensor_(getri)(THTensor *ra_, THTensor *a);

#endif

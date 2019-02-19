#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorAssignments.h"
#else

TH_API void THTensor_(fill)(THTensor *r_, scalar_t value);
TH_API void THTensor_(zero)(THTensor *r_);

TH_API void THTensor_(eye)(THTensor *r_, int64_t n, int64_t m);
#endif

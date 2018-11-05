#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorCopy.h"
#else

/* Support for copy between different Tensor types */

TH_API void THTensor_(copy)(THTensor *tensor, THTensor *src);

#endif

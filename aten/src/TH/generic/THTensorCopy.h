#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorCopy.h"
#else

/* Support for copy between different Tensor types */

TH_API void THTensor_(copy)(THTensor *tensor, THTensor *src);
TH_API void THTensor_(copyByte)(THTensor *tensor, struct THByteTensor *src);
TH_API void THTensor_(copyChar)(THTensor *tensor, struct THCharTensor *src);
TH_API void THTensor_(copyShort)(THTensor *tensor, struct THShortTensor *src);
TH_API void THTensor_(copyInt)(THTensor *tensor, struct THIntTensor *src);
TH_API void THTensor_(copyLong)(THTensor *tensor, struct THLongTensor *src);
TH_API void THTensor_(copyFloat)(THTensor *tensor, struct THFloatTensor *src);
TH_API void THTensor_(copyDouble)(THTensor *tensor, struct THDoubleTensor *src);
TH_API void THTensor_(copyHalf)(THTensor *tensor, struct THHalfTensor *src);

#endif

#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZTensorCopy.h"
#else

/* Support for copy between different Tensor types */

TH_API void THZTensor_(copy)(THZTensor *tensor, THZTensor *src);
TH_API void THZTensor_(copyByte)(THZTensor *tensor, struct THByteTensor *src);
TH_API void THZTensor_(copyChar)(THZTensor *tensor, struct THCharTensor *src);
TH_API void THZTensor_(copyShort)(THZTensor *tensor, struct THShortTensor *src);
TH_API void THZTensor_(copyInt)(THZTensor *tensor, struct THIntTensor *src);
TH_API void THZTensor_(copyLong)(THZTensor *tensor, struct THLongTensor *src);
TH_API void THZTensor_(copyFloat)(THZTensor *tensor, struct THFloatTensor *src);
TH_API void THZTensor_(copyDouble)(THZTensor *tensor, struct THDoubleTensor *src);
TH_API void THZTensor_(copyZFloat)(THZTensor *tensor, struct THZFloatTensor *src);
TH_API void THZTensor_(copyZDouble)(THZTensor *tensor, struct THZDoubleTensor *src);
TH_API void THZTensor_(copyHalf)(THZTensor *tensor, struct THHalfTensor *src);

#endif

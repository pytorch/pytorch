#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THStorageCopy.h"
#else

/* Support for copy between different Storage types */

TH_API void THStorage_(rawCopy)(THStorage *storage, scalar_t *src);
TH_API void THStorage_(copy)(THStorage *storage, THStorage *src);
TH_API void THStorage_(copyByte)(THStorage *storage, struct THByteStorage *src);
TH_API void THStorage_(copyChar)(THStorage *storage, struct THCharStorage *src);
TH_API void THStorage_(copyShort)(THStorage *storage, struct THShortStorage *src);
TH_API void THStorage_(copyInt)(THStorage *storage, struct THIntStorage *src);
TH_API void THStorage_(copyLong)(THStorage *storage, struct THLongStorage *src);
TH_API void THStorage_(copyFloat)(THStorage *storage, struct THFloatStorage *src);
TH_API void THStorage_(copyDouble)(THStorage *storage, struct THDoubleStorage *src);
TH_API void THStorage_(copyHalf)(THStorage *storage, struct THHalfStorage *src);
TH_API void THStorage_(copyBool)(THStorage *storage, struct THBoolStorage *src);
TH_API void THStorage_(copyBFloat16)(THStorage *storage, struct THBFloat16Storage *src);

#endif

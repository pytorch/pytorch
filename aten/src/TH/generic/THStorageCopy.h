#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THStorageCopy.h"
#else

/* Support for copy between different Storage types */

TH_API void THStorage_(rawCopy)(THStorage *storage, real *src);
TH_API void THStorage_(copy)(THStorage *storage, THStorage *src);
TH_API void THStorage_(copyByte)(THStorage *storage, struct THByteStorage *src);
TH_API void THStorage_(copyChar)(THStorage *storage, struct THCharStorage *src);
TH_API void THStorage_(copyShort)(THStorage *storage, struct THShortStorage *src);
TH_API void THStorage_(copyInt)(THStorage *storage, struct THIntStorage *src);
TH_API void THStorage_(copyLong)(THStorage *storage, struct THLongStorage *src);
TH_API void THStorage_(copyFloat)(THStorage *storage, struct THFloatStorage *src);
TH_API void THStorage_(copyDouble)(THStorage *storage, struct THDoubleStorage *src);
TH_API void THStorage_(copyHalf)(THStorage *storage, struct THHalfStorage *src);

#endif

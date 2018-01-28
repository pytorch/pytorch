#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZStorageCopy.h"
#else

/* Support for copy between different Storage types */

TH_API void THZStorage_(rawCopy)(THZStorage *storage, ntype *src);
TH_API void THZStorage_(copy)(THZStorage *storage, THZStorage *src);
TH_API void THZStorage_(copyByte)(THZStorage *storage, struct THByteStorage *src);
TH_API void THZStorage_(copyChar)(THZStorage *storage, struct THCharStorage *src);
TH_API void THZStorage_(copyShort)(THZStorage *storage, struct THShortStorage *src);
TH_API void THZStorage_(copyInt)(THZStorage *storage, struct THIntStorage *src);
TH_API void THZStorage_(copyLong)(THZStorage *storage, struct THLongStorage *src);
TH_API void THZStorage_(copyFloat)(THZStorage *storage, struct THFloatStorage *src);
TH_API void THZStorage_(copyDouble)(THZStorage *storage, struct THDoubleStorage *src);
TH_API void THZStorage_(copyZFloat)(THZStorage *storage, struct THZFloatStorage *src);
TH_API void THZStorage_(copyZDouble)(THZStorage *storage, struct THZDoubleStorage *src);
TH_API void THZStorage_(copyHalf)(THZStorage *storage, struct THHalfStorage *src);

#endif

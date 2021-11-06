#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THStorageCopy.h"
#else

/* Support for copy between different Storage types */
TH_API void THStorage_(copy)(THStorage *storage, THStorage *src);

TH_API void THStorage_(copyByte)(THStorage *storage, struct THByteStorage *src);

#endif

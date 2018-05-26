#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THStorageCopy.h"
#else

/* Support for copy between different Storage types */

TH_API void THStorage_(rawCopy)(at::StorageImpl *storage, real *src);
TH_API void THStorage_(copy)(at::StorageImpl *storage, at::StorageImpl *src);
TH_API void THStorage_(copyByte)(at::StorageImpl *storage, at::ByteStorageImpl *src);
TH_API void THStorage_(copyChar)(at::StorageImpl *storage, at::CharStorageImpl *src);
TH_API void THStorage_(copyShort)(at::StorageImpl *storage, at::ShortStorageImpl *src);
TH_API void THStorage_(copyInt)(at::StorageImpl *storage, at::IntStorageImpl *src);
TH_API void THStorage_(copyLong)(at::StorageImpl *storage, at::LongStorageImpl *src);
TH_API void THStorage_(copyFloat)(at::StorageImpl *storage, at::FloatStorageImpl *src);
TH_API void THStorage_(copyDouble)(at::StorageImpl *storage, at::DoubleStorageImpl *src);
TH_API void THStorage_(copyHalf)(at::StorageImpl *storage, at::HalfStorageImpl *src);

#endif

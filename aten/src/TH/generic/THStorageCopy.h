#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THStorageCopy.h"
#else

/* Support for copy between different Storage types */
TH_API void THStorage_(copy)(THStorage *storage, THStorage *src);
#ifdef TH_REAL_IS_BYTE
  TH_API void THStorage_(copyByte)(THStorage *storage, struct THByteStorage *src);
#elif defined(TH_REAL_IS_CHAR)
  TH_API void THStorage_(copyChar)(THStorage *storage, struct THCharStorage *src);
#elif defined(TH_REAL_IS_SHORT)
  TH_API void THStorage_(copyShort)(THStorage *storage, struct THShortStorage *src);
#elif defined(TH_REAL_IS_INT)
  TH_API void THStorage_(copyInt)(THStorage *storage, struct THIntStorage *src);
#elif defined(TH_REAL_IS_LONG)
  TH_API void THStorage_(copyLong)(THStorage *storage, struct THLongStorage *src);
#elif defined(TH_REAL_IS_FLOAT)
  TH_API void THStorage_(copyFloat)(THStorage *storage, struct THFloatStorage *src);
#elif defined(TH_REAL_IS_DOUBLE)
  TH_API void THStorage_(copyDouble)(THStorage *storage, struct THDoubleStorage *src);
#elif defined(TH_REAL_IS_HALF)
  TH_API void THStorage_(copyHalf)(THStorage *storage, struct THHalfStorage *src);
#elif defined(TH_REAL_IS_BOOL)
  TH_API void THStorage_(copyBool)(THStorage *storage, struct THBoolStorage *src);
#elif defined(TH_REAL_IS_BFLOAT16)
  TH_API void THStorage_(copyBFloat16)(THStorage *storage, struct THBFloat16Storage *src);
#elif defined(THQUANTIZED)
  #ifdef THQUINT8
  TH_API void THStorage_(copyQUInt8)(THStorage *storage, struct THQUInt8Storage *src);
  #elif defined(THQINT8)
  TH_API void THStorage_(copyQInt8)(THStorage *storage, struct THQInt8Storage *src);
  #elif defined(THQINT32)
  TH_API void THStorage_(copyQInt32)(THStorage *storage, struct THQInt32Storage *src);
  #endif
#endif

#endif

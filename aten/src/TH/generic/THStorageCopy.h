#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THStorageCopy.h"
#else

/* Support for copy between different Storage types */
TH_API void THStorage_(copy)(THStorage *storage, THStorage *src);

// TODO: Add cross-dtype storage copy for complex storage
#if !defined(TH_REAL_IS_COMPLEXFLOAT) && !defined(TH_REAL_IS_COMPLEXDOUBLE)
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
  #ifdef THQUINT8
    TH_API void THStorage_(copyQUInt8)(THStorage *storage, struct THQUInt8Storage *src);
  #endif
  #ifdef THQINT8
    TH_API void THStorage_(copyQInt8)(THStorage *storage, struct THQInt8Storage *src);
  #endif
  #ifdef THQINT32
    TH_API void THStorage_(copyQInt32)(THStorage *storage, struct THQInt32Storage *src);
  #endif
#else
  TH_API void THStorage_(copyComplexFloat)(THStorage *storage, struct THComplexFloatStorage *src);
  TH_API void THStorage_(copyComplexDouble)(THStorage *storage, struct THComplexDoubleStorage *src);
#endif

#endif

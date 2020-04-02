#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THStorageCopy.cpp"
#else

void THStorage_(copy)(THStorage *storage, THStorage *src)
{
  THArgCheck(storage->numel() == src->numel(), 2, "size mismatch");
  scalar_t *scalar_src = THStorage_(data)(src);
  scalar_t *data = THStorage_(data)(storage);
  for (ptrdiff_t i = 0; i < storage->numel(); ++i) {
    data[i] = scalar_src[i];
  }
}

// NOTE: for performance, these macros generally use the raw data pointer in the inner loops,
// rather than repeated THStorage_(data) calls.

#define IMPLEMENT_THStorage_COPY(TYPENAMESRC) \
void THStorage_(copy##TYPENAMESRC)(THStorage *storage, TH##TYPENAMESRC##Storage *src) \
{ \
  ptrdiff_t i;                                                          \
  auto data = THStorage_(data)(storage);                                \
  auto src_data = TH##TYPENAMESRC##Storage_data(src);                   \
  for(i = 0; i < storage->numel(); i++)                                    \
    data[i] = static_cast<scalar_t>(src_data[i]);                           \
}

#ifdef TH_REAL_IS_BYTE
  IMPLEMENT_THStorage_COPY(Byte)
#elif defined(TH_REAL_IS_CHAR)
  IMPLEMENT_THStorage_COPY(Char)
#elif defined(TH_REAL_IS_SHORT)
  IMPLEMENT_THStorage_COPY(Short)
#elif defined(TH_REAL_IS_INT)
  IMPLEMENT_THStorage_COPY(Int)
#elif defined(TH_REAL_IS_LONG)
  IMPLEMENT_THStorage_COPY(Long)
#elif defined(TH_REAL_IS_FLOAT)
  IMPLEMENT_THStorage_COPY(Float)
#elif defined(TH_REAL_IS_DOUBLE)
  IMPLEMENT_THStorage_COPY(Double)
#elif defined(TH_REAL_IS_HALF)
  IMPLEMENT_THStorage_COPY(Half)
#elif defined(TH_REAL_IS_BOOL)
  IMPLEMENT_THStorage_COPY(Bool)
#elif defined(TH_REAL_IS_BFLOAT16)
  IMPLEMENT_THStorage_COPY(BFloat16)
#elif defined(THQUANTIZED)
  #ifdef THQUINT8
  IMPLEMENT_THStorage_COPY(QUInt8)
  #endif
  #ifdef THQINT8
  IMPLEMENT_THStorage_COPY(QInt8)
  #endif
  #ifdef THQINT32
  IMPLEMENT_THStorage_COPY(QInt32)
  #endif
#endif

#endif

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

IMPLEMENT_THStorage_COPY(Byte)
IMPLEMENT_THStorage_COPY(Char)
IMPLEMENT_THStorage_COPY(Short)
IMPLEMENT_THStorage_COPY(Int)
IMPLEMENT_THStorage_COPY(Long)
IMPLEMENT_THStorage_COPY(Float)
IMPLEMENT_THStorage_COPY(Double)
IMPLEMENT_THStorage_COPY(Half)
IMPLEMENT_THStorage_COPY(Bool)
IMPLEMENT_THStorage_COPY(BFloat16)
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

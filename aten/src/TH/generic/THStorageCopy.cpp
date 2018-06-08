#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THStorageCopy.cpp"
#else

void THStorage_(rawCopy)(THStorage *storage, real *src)
{
  ptrdiff_t i;
  real *data = THStorage_(data)(storage);
  for(i = 0; i < storage->size; i++)
    data[i] = src[i];
}

void THStorage_(copy)(THStorage *storage, THStorage *src)
{
  THArgCheck(storage->size == src->size, 2, "size mismatch");
  THStorage_(rawCopy)(storage, THStorage_(data)(src));
}

// NOTE: for performance, these macros generally use the raw data pointer in the inner loops,
// rather than repeated THStorage_(data) calls.

#define IMPLEMENT_THStorage_COPY(TYPENAMESRC) \
void THStorage_(copy##TYPENAMESRC)(THStorage *storage, TH##TYPENAMESRC##Storage *src) \
{ \
  ptrdiff_t i;                                                          \
  auto data = THStorage_(data)(storage);                                \
  auto src_data = TH##TYPENAMESRC##Storage_data(src);                   \
  for(i = 0; i < storage->size; i++)                                    \
    data[i] = static_cast<real>(src_data[i]);                           \
}

#define IMPLEMENT_THStorage_COPY_FROM_HALF(TYPENAMESRC)		\
void THStorage_(copy##TYPENAMESRC)(THStorage *storage, TH##TYPENAMESRC##Storage *src) \
{ \
  THArgCheck(storage->size == src->size, 2, "size mismatch"); \
  ptrdiff_t i;								\
  auto data = THStorage_(data)(storage);      \
  auto src_data = TH##TYPENAMESRC##Storage_data(src); \
  for(i = 0; i < storage->size; i++)					\
    data[i] = (real)TH_half2float(src_data[i]); \
}

#define IMPLEMENT_THStorage_COPY_TO_HALF(TYPENAMESRC)		\
void THStorage_(copy##TYPENAMESRC)(THStorage *storage, TH##TYPENAMESRC##Storage *src) \
{ \
  THArgCheck(storage->size == src->size, 2, "size mismatch"); \
  ptrdiff_t i;								\
  auto data = THStorage_(data)(storage);      \
  auto src_data = TH##TYPENAMESRC##Storage_data(src); \
  for(i = 0; i < storage->size; i++)					\
    data[i] = TH_float2half((float)(src_data[i])); \
}

#define IMPLEMENT_THStorage_COPY_TO_FROM_HALF(TYPENAMESRC)		\
void THStorage_(copy##TYPENAMESRC)(THStorage *storage, TH##TYPENAMESRC##Storage *src) \
{ \
  THArgCheck(storage->size == src->size, 2, "size mismatch"); \
  ptrdiff_t i;								\
  auto data = THStorage_(data)(storage);      \
  auto src_data = TH##TYPENAMESRC##Storage_data(src); \
  for(i = 0; i < storage->size; i++)					\
    data[i] = static_cast<real>(src_data[i]); \
}

#ifndef TH_REAL_IS_HALF
IMPLEMENT_THStorage_COPY(Byte)
IMPLEMENT_THStorage_COPY(Char)
IMPLEMENT_THStorage_COPY(Short)
IMPLEMENT_THStorage_COPY(Int)
IMPLEMENT_THStorage_COPY(Long)
IMPLEMENT_THStorage_COPY(Float)
IMPLEMENT_THStorage_COPY(Double)
IMPLEMENT_THStorage_COPY_FROM_HALF(Half)
#else
/* only allow pass-through for Half */
IMPLEMENT_THStorage_COPY_TO_FROM_HALF(Half)
IMPLEMENT_THStorage_COPY_TO_HALF(Byte)
IMPLEMENT_THStorage_COPY_TO_HALF(Char)
IMPLEMENT_THStorage_COPY_TO_HALF(Short)
IMPLEMENT_THStorage_COPY_TO_HALF(Int)
IMPLEMENT_THStorage_COPY_TO_HALF(Long)
IMPLEMENT_THStorage_COPY_TO_HALF(Float)
IMPLEMENT_THStorage_COPY_TO_HALF(Double)
#endif


#endif

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THStorageCopy.cpp"
#else

void THStorage_(rawCopy)(at::StorageImpl *storage, real *src)
{
  ptrdiff_t i;
  for(i = 0; i < storage->size(); i++)
    storage->data<real>()[i] = src[i];
}

void THStorage_(copy)(at::StorageImpl *storage, at::StorageImpl *src)
{
  THArgCheck(storage->size() == src->size(), 2, "size mismatch");
  THStorage_(rawCopy)(storage, src->data<real>());
}

#define IMPLEMENT_THStorage_COPY(TYPENAMESRC) \
void THStorage_(copy##TYPENAMESRC)(at::StorageImpl *storage, at::TYPENAMESRC##StorageImpl *src) \
{ \
  ptrdiff_t i;                                                        \
  for(i = 0; i < storage->size(); i++)                                  \
    storage->data<real>()[i] = static_cast<real>(TH##TYPENAMESRC##Storage_data(src)[i]);           \
}

#define IMPLEMENT_THStorage_COPY_FROM_HALF(TYPENAMESRC)		\
void THStorage_(copy##TYPENAMESRC)(at::StorageImpl *storage, at::TYPENAMESRC##StorageImpl *src) \
{ \
  THArgCheck(storage->size() == src->size(), 2, "size mismatch"); \
  ptrdiff_t i;								\
  for(i = 0; i < storage->size(); i++)					\
    storage->data<real>()[i] = (real)TH_half2float(TH##TYPENAMESRC##Storage_data(src)[i]);		\
}

#define IMPLEMENT_THStorage_COPY_TO_HALF(TYPENAMESRC)		\
void THStorage_(copy##TYPENAMESRC)(at::StorageImpl *storage, at::TYPENAMESRC##StorageImpl *src) \
{ \
  THArgCheck(storage->size() == src->size(), 2, "size mismatch"); \
  ptrdiff_t i;								\
  for(i = 0; i < storage->size(); i++)					\
    storage->data<real>()[i] = TH_float2half((float)(TH##TYPENAMESRC##Storage_data(src)[i]));		\
}

#define IMPLEMENT_THStorage_COPY_TO_FROM_HALF(TYPENAMESRC)		\
void THStorage_(copy##TYPENAMESRC)(at::StorageImpl *storage, at::TYPENAMESRC##StorageImpl *src) \
{ \
  THArgCheck(storage->size() == src->size(), 2, "size mismatch"); \
  ptrdiff_t i;								\
  for(i = 0; i < storage->size(); i++)					\
    storage->data<real>()[i] = static_cast<real>(TH##TYPENAMESRC##Storage_data(src)[i]);		\
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

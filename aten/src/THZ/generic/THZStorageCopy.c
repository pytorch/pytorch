#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZStorageCopy.c"
#else

void THZStorage_(rawCopy)(THZStorage *storage, ntype *src)
{
  ptrdiff_t i;
  for(i = 0; i < storage->size; i++)
    storage->data[i] = src[i];
}

void THZStorage_(copy)(THZStorage *storage, THZStorage *src)
{
  THArgCheck(storage->size == src->size, 2, "size mismatch");
  THZStorage_(rawCopy)(storage, src->data);
}

#define IMPLEMENT_THZStorage_COPY(TYPENAMESRC) \
void THZStorage_(copy##TYPENAMESRC)(THZStorage *storage, TH##TYPENAMESRC##Storage *src) \
{ \
  ptrdiff_t i;                                                        \
  for(i = 0; i < storage->size; i++)                                  \
    storage->data[i] = (ntype)src->data[i];                            \
}

#define IMPLEMENT_THZStorage_COPY_FROM_HALF(TYPENAMESRC)   \
void THZStorage_(copy##TYPENAMESRC)(THZStorage *storage, TH##TYPENAMESRC##Storage *src) \
{ \
  THArgCheck(storage->size == src->size, 2, "size mismatch"); \
  ptrdiff_t i;                \
  for(i = 0; i < storage->size; i++)          \
    storage->data[i] = (ntype)TH_half2float(src->data[i]);    \
}

#define IMPLEMENT_THZStorage_COPY_TO_HALF(TYPENAMESRC)   \
void THZStorage_(copy##TYPENAMESRC)(THZStorage *storage, TH##TYPENAMESRC##Storage *src) \
{ \
  THArgCheck(storage->size == src->size, 2, "size mismatch"); \
  ptrdiff_t i;                \
  for(i = 0; i < storage->size; i++)          \
    storage->data[i] = TH_float2half((float)(src->data[i]));    \
}

#define IMPLEMENT_THZStorage_COPY_TO_FROM_HALF(TYPENAMESRC)    \
void THZStorage_(copy##TYPENAMESRC)(THZStorage *storage, TH##TYPENAMESRC##Storage *src) \
{ \
  THArgCheck(storage->size == src->size, 2, "size mismatch"); \
  ptrdiff_t i;                \
  for(i = 0; i < storage->size; i++)          \
    storage->data[i] = src->data[i];    \
}

/* direct copy from a complex type will only copy its real part
 * which is based on C's complex number type cast behaviour;
 */

#ifndef TH_NTYPE_IS_HALF
IMPLEMENT_THZStorage_COPY(Byte)
IMPLEMENT_THZStorage_COPY(Char)
IMPLEMENT_THZStorage_COPY(Short)
IMPLEMENT_THZStorage_COPY(Int)
IMPLEMENT_THZStorage_COPY(Long)
IMPLEMENT_THZStorage_COPY(Float)
IMPLEMENT_THZStorage_COPY(Double)
IMPLEMENT_THZStorage_COPY(ZFloat)
IMPLEMENT_THZStorage_COPY(ZDouble)
IMPLEMENT_THZStorage_COPY_FROM_HALF(Half)
#else
/* only allow pass-through for Half */
IMPLEMENT_THZStorage_COPY_TO_FROM_HALF(Half)
IMPLEMENT_THZStorage_COPY_TO_HALF(Byte)
IMPLEMENT_THZStorage_COPY_TO_HALF(Char)
IMPLEMENT_THZStorage_COPY_TO_HALF(Short)
IMPLEMENT_THZStorage_COPY_TO_HALF(Int)
IMPLEMENT_THZStorage_COPY_TO_HALF(Long)
IMPLEMENT_THZStorage_COPY_TO_HALF(Float)
IMPLEMENT_THZStorage_COPY_TO_HALF(Double)
IMPLEMENT_THZStorage_COPY_TO_HALF(ZFloat)
IMPLEMENT_THZStorage_COPY_TO_HALF(ZDouble)
#endif

#endif

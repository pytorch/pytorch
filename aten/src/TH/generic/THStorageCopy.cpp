#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THStorageCopy.cpp"
#else

void THStorage_(copy)(THStorage *storage, THStorage *src)
{
  THArgCheck(storage->nbytes() == src->nbytes(), 2, "size mismatch");
  scalar_t *scalar_src = THStorage_(data)(src);
  scalar_t *data = THStorage_(data)(storage);
  uint64_t numel = storage->nbytes() / sizeof(scalar_t);
  for (uint64_t i = 0; i < numel; ++i) {
    data[i] = scalar_src[i];
  }
}

// NOTE: for performance, these macros generally use the raw data pointer in the inner loops,
// rather than repeated THStorage_(data) calls.

#define IMPLEMENT_THStorage_COPY(TYPENAMESRC)                \
  void THStorage_(copy##TYPENAMESRC)(                        \
      THStorage * storage, TH##TYPENAMESRC##Storage * src) { \
    auto data = THStorage_(data)(storage);                   \
    auto src_data = TH##TYPENAMESRC##Storage_data(src);      \
    uint64_t numel = storage->nbytes() / sizeof(scalar_t);   \
    for (uint64_t i = 0; i < numel; i++)                     \
      data[i] = static_cast<scalar_t>(src_data[i]);          \
  }

  IMPLEMENT_THStorage_COPY(Byte)

#endif

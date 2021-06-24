#include "miniz.h"
#include <iostream>

#include "caffe2/serialize/crc_alt.h"

extern "C" {
// See: miniz.h
#if defined(USE_EXTERNAL_MZCRC)
mz_ulong mz_crc32(mz_ulong crc, const mz_uint8* ptr, size_t buf_len) {
  auto z = crc32_fast(ptr, buf_len, crc);
  return z;
};
#endif
}

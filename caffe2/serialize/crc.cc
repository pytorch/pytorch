#include "miniz.h"
#include <iostream>

extern "C" {
mz_ulong mz_crc32(mz_ulong crc, const mz_uint8* ptr, size_t buf_len) {
  // std::cout << "Custom crc!\n";
  return 0;
}
}

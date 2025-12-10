#ifndef __CUDACC__
#  error "A C or C++ compiler has been selected for CUDA"
#endif

#include "CMakeCompilerABI.h"
#include "CMakeCompilerCUDAArch.h"

int main(int argc, char* argv[])
{
  int require = 0;
  require += info_sizeof_dptr[argc];
  require += info_byte_order_big_endian[argc];
  require += info_byte_order_little_endian[argc];
#if defined(ABI_ID)
  require += info_abi[argc];
#endif
#if defined(ARCHITECTURE_ID)
  require += info_arch[argc];
#endif
  static_cast<void>(argv);

  if (!cmakeCompilerCUDAArch()) {
    // Convince the compiler that the non-zero return value depends
    // on the info strings so they are not optimized out.
    return require ? -1 : 1;
  }

  return 0;
}

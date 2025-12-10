#if !defined(__HIP__) && !defined(__NVCC__)
#  error "A C or C++ compiler has been selected for HIP"
#endif

#include "CMakeCompilerABI.h"

#if defined(__NVCC__)
#  include "CMakeCompilerCUDAArch.h"
#endif

int main(int argc, char* argv[])
{
  int require = 0;
  require += info_sizeof_dptr[argc];
#if defined(ABI_ID)
  require += info_abi[argc];
#endif
#if defined(ARCHITECTURE_ID)
  require += info_arch[argc];
#endif
  static_cast<void>(argv);

#if defined(__NVCC__)
  if (!cmakeCompilerCUDAArch()) {
    // Convince the compiler that the non-zero return value depends
    // on the info strings so they are not optimized out.
    return require ? -1 : 1;
  }
  return 0;
#else
  return require;
#endif
}

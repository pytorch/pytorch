#ifndef __cplusplus
# error "A C compiler has been selected for Objective-C++."
#endif

/*--------------------------------------------------------------------------*/

#include "CMakeCompilerABI.h"

/*--------------------------------------------------------------------------*/

int main(int argc, char *argv[])
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
  (void)argv;
  return require;
}

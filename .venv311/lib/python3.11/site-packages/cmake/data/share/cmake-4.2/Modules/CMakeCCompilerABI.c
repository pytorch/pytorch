#ifdef __cplusplus
#  error "A C++ compiler has been selected for C."
#endif

#ifdef __CLASSIC_C__
#  define const
#endif

#include "CMakeCompilerABI.h"

#ifdef __CLASSIC_C__
int main(argc, argv)
int argc;
char* argv[];
#else
int main(int argc, char* argv[])
#endif
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


/* Size of a pointer-to-data in bytes.  */
#define SIZEOF_DPTR (sizeof(void*))
char const info_sizeof_dptr[] = {
  /* clang-format off */
  'I', 'N', 'F', 'O', ':', 's', 'i', 'z', 'e', 'o', 'f', '_', 'd', 'p', 't',
  'r', '[', ('0' + ((SIZEOF_DPTR / 10) % 10)), ('0' + (SIZEOF_DPTR % 10)), ']',
  '\0'
  /* clang-format on */
};

/* Byte order.  Only one of these will have bytes in the right order.  */
static unsigned short const info_byte_order_big_endian[] = {
  /* INFO:byte_order string for BIG_ENDIAN */
  0x494E, 0x464F, 0x3A62, 0x7974, 0x655F, 0x6F72, 0x6465, 0x725B,
  0x4249, 0x475F, 0x454E, 0x4449, 0x414E, 0x5D00, 0x0000
};
static unsigned short const info_byte_order_little_endian[] = {
  /* INFO:byte_order string for LITTLE_ENDIAN */
  0x4E49, 0x4F46, 0x623A, 0x7479, 0x5F65, 0x726F, 0x6564, 0x5B72,
  0x494C, 0x5454, 0x454C, 0x455F, 0x444E, 0x4149, 0x5D4E, 0x0000
};

/* Application Binary Interface.  */

/* Check for (some) ARM ABIs.
 * See e.g. http://wiki.debian.org/ArmEabiPort for some information on this. */
#if defined(__GNU__) && defined(__ELF__) && defined(__ARM_EABI__)
#  define ABI_ID "ELF ARMEABI"
#elif defined(__GNU__) && defined(__ELF__) && defined(__ARMEB__)
#  define ABI_ID "ELF ARM"
#elif defined(__GNU__) && defined(__ELF__) && defined(__ARMEL__)
#  define ABI_ID "ELF ARM"

#elif defined(__linux__) && defined(__ELF__) && defined(__amd64__) &&         \
  defined(__ILP32__)
#  define ABI_ID "ELF X32"

#elif defined(__ELF__)
#  define ABI_ID "ELF"
#endif

/* Sync with:
 *   Help/variable/CMAKE_LANG_COMPILER_ARCHITECTURE_ID.rst
 *   Modules/CMakeFortranCompilerABI.F
 *   Modules/CMakeFortranCompilerABI.F90
 *   Modules/Internal/CMakeParseCompilerArchitectureId.cmake
 */
#if defined(__APPLE__) && defined(__arm64__)
#  if defined(__ARM64_ARCH_8_32__)
#    define ARCHITECTURE_ID "arm64_32"
#  elif defined(__arm64e__)
#    define ARCHITECTURE_ID "arm64e"
#  else
#    define ARCHITECTURE_ID "arm64"
#  endif
#elif defined(_MSC_VER) && defined(_M_ARM64EC)
#  define ARCHITECTURE_ID "arm64ec"
#elif defined(_MSC_VER) && defined(_M_ARM64)
#  define ARCHITECTURE_ID "arm64"
#elif defined(__arm64ec__)
#  define ARCHITECTURE_ID "arm64ec"
#elif defined(__aarch64__)
#  define ARCHITECTURE_ID "aarch64"
#elif __ARM_ARCH == 7 || _M_ARM == 7 || defined(__ARM_ARCH_7__) ||            \
  defined(__TI_ARM_V7__)
#  if defined(__APPLE__) && defined(__ARM_ARCH_7K__)
#    define ARCHITECTURE_ID "armv7k"
#  elif defined(__APPLE__) && defined(__ARM_ARCH_7S__)
#    define ARCHITECTURE_ID "armv7s"
#  else
#    define ARCHITECTURE_ID "armv7"
#  endif
#elif __ARM_ARCH == 6 || _M_ARM == 6 || defined(__ARM_ARCH_6__) ||            \
  defined(__TI_ARM_V6__)
#  define ARCHITECTURE_ID "armv6"
#elif __ARM_ARCH == 5 || _M_ARM == 5 || defined(__ARM_ARCH_5__) ||            \
  defined(__TI_ARM_V5__)
#  define ARCHITECTURE_ID "armv5"
#elif defined(__alpha) || defined(__alpha) || defined(_M_ALPHA)
#  define ARCHITECTURE_ID "alpha"
#elif defined(__x86_64) || defined(__x86_64__) || defined(__amd64) ||         \
  defined(__amd64__) || defined(_M_X64) || defined(_M_AMD64)
#  define ARCHITECTURE_ID "x86_64"
#elif defined(__i686) || defined(__i686__) || _M_IX86 == 600
#  define ARCHITECTURE_ID "i686"
#elif defined(__i586) || defined(__i586__) || _M_IX86 == 500
#  define ARCHITECTURE_ID "i586"
#elif defined(__i486) || defined(__i486__) || _M_IX86 == 400
#  define ARCHITECTURE_ID "i486"
#elif defined(__i386) || defined(__i386__) || defined(_M_IX86)
#  define ARCHITECTURE_ID "i386"
#elif defined(__ia64) || defined(__ia64__) || defined(_M_IA64)
#  define ARCHITECTURE_ID "ia64"
#elif defined(__loongarch64)
#  define ARCHITECTURE_ID "loongarch64"
#elif defined(__loongarch__)
#  define ARCHITECTURE_ID "loongarch32"
#elif defined(__m68k__)
#  define ARCHITECTURE_ID "m68k"
#elif defined(__mips64) || defined(__mips64__)
#  if defined(_MIPSEL)
#    define ARCHITECTURE_ID "mips64el"
#  else
#    define ARCHITECTURE_ID "mips64"
#  endif
#elif defined(__mips) || defined(__mips__)
#  if defined(_MIPSEL)
#    define ARCHITECTURE_ID "mipsel"
#  else
#    define ARCHITECTURE_ID "mips"
#  endif
#elif defined(__riscv) && __riscv_xlen == 64
#  define ARCHITECTURE_ID "riscv64"
#elif defined(__riscv) && __riscv_xlen == 32
#  define ARCHITECTURE_ID "riscv32"
#elif defined(__sw_64)
#  define ARCHITECTURE_ID "sw_64"
#elif defined(__s390x__)
#  define ARCHITECTURE_ID "s390x"
#elif defined(__s390__)
#  define ARCHITECTURE_ID "s390"
#elif defined(__sparcv9) || defined(__sparcv9__) || defined(__sparc64__)
#  define ARCHITECTURE_ID "sparcv9"
#elif defined(__sparc) || defined(__sparc__)
#  define ARCHITECTURE_ID "sparc"
#elif defined(__hppa) || defined(__hppa__)
#  if defined(__LP64__)
#    define ARCHITECTURE_ID "parisc64"
#  else
#    define ARCHITECTURE_ID "parisc"
#  endif
#elif defined(__ppc64__) || defined(__powerpc64__) || defined(__PPC64__) ||   \
  defined(_ARCH_PPC64)
#  if defined(_LITTLE_ENDIAN) || defined(__LITTLE_ENDIAN__)
#    define ARCHITECTURE_ID "ppc64le"
#  else
#    define ARCHITECTURE_ID "ppc64"
#  endif
#elif defined(__ppc__) || defined(__powerpc__) || defined(__PPC__) ||         \
  defined(_ARCH_PPC)
#  if defined(_LITTLE_ENDIAN) || defined(__LITTLE_ENDIAN__)
#    define ARCHITECTURE_ID "ppcle"
#  else
#    define ARCHITECTURE_ID "ppc"
#  endif
#endif

/* Construct the string literal in pieces to prevent the source from
   getting matched.  Store it in a pointer rather than an array
   because some compilers will just produce instructions to fill the
   array rather than assigning a pointer to a static array.  */
#if defined(ABI_ID)
static char const* info_abi = "INFO"
                              ":"
                              "abi[" ABI_ID "]";
#endif
#if defined(ARCHITECTURE_ID)
static char const* info_arch = "INFO"
                               ":"
                               "arch[" ARCHITECTURE_ID "]";
#endif

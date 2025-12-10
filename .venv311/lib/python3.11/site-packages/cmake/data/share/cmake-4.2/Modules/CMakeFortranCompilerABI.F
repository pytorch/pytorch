      PROGRAM CMakeFortranCompilerABI
#if 0
! Address Size
#endif
#if defined(_LP64)
        PRINT *, 'INFO:sizeof_dptr[8]'
#elif defined(_M_IA64)
        PRINT *, 'INFO:sizeof_dptr[8]'
#elif defined(_M_X64)
        PRINT *, 'INFO:sizeof_dptr[8]'
#elif defined(_M_AMD64)
        PRINT *, 'INFO:sizeof_dptr[8]'
#elif defined(__x86_64__)
        PRINT *, 'INFO:sizeof_dptr[8]'
#elif defined(__sparcv9) || defined(__sparcv9__) || defined(__sparc64__)
        PRINT *, 'INFO:sizeof_dptr[8]'

#elif defined(_ILP32)
        PRINT *, 'INFO:sizeof_dptr[4]'
#elif defined(_M_IX86)
        PRINT *, 'INFO:sizeof_dptr[4]'
#elif defined(__i386__)
        PRINT *, 'INFO:sizeof_dptr[4]'
#elif defined(__sparc) || defined(__sparc__)
        PRINT *, 'INFO:sizeof_dptr[4]'

#elif defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 8
        PRINT *, 'INFO:sizeof_dptr[8]'
#elif defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 4
        PRINT *, 'INFO:sizeof_dptr[4]'
#elif defined(__SIZEOF_SIZE_T__) && __SIZEOF_SIZE_T__ == 8
        PRINT *, 'INFO:sizeof_dptr[8]'
#elif defined(__SIZEOF_SIZE_T__) && __SIZEOF_SIZE_T__ == 4
        PRINT *, 'INFO:sizeof_dptr[4]'
#endif

#if 0
! Application Binary Interface
#endif
#if defined(__ELF__)
        PRINT *, 'INFO:abi[ELF]'
#endif

#if 0
! Sync with
!   Help/variable/CMAKE_LANG_COMPILER_ARCHITECTURE_ID.rst
!   Modules/CMakeCompilerABI.h
!   Modules/CMakeFortranCompilerABI.F90
!   Modules/Internal/CMakeParseCompilerArchitectureId.cmake
#endif
#if defined(__APPLE__) && defined(__arm64__)
#  if defined(__ARM64_ARCH_8_32__)
        PRINT *, 'INFO:arch[arm64_32]'
#  elif defined(__arm64e__)
        PRINT *, 'INFO:arch[arm64e]'
#  else
        PRINT *, 'INFO:arch[arm64]'
#  endif
#elif defined(_MSC_VER) && defined(_M_ARM64EC)
        PRINT *, 'INFO:arch[arm64ec]'
#elif defined(_MSC_VER) && defined(_M_ARM64)
        PRINT *, 'INFO:arch[arm64]'
#elif defined(__arm64ec__)
        PRINT *, 'INFO:arch[arm64ec]'
#elif defined(__aarch64__)
        PRINT *, 'INFO:arch[aarch64]'
#elif __ARM_ARCH == 7 || _M_ARM == 7 || defined(__ARM_ARCH_7__)
#  if defined(__APPLE__) && defined(__ARM_ARCH_7K__)
        PRINT *, 'INFO:arch[armv7k]'
#  elif defined(__APPLE__) && defined(__ARM_ARCH_7S__)
        PRINT *, 'INFO:arch[armv7s]'
#  else
        PRINT *, 'INFO:arch[armv7]'
#  endif
#elif __ARM_ARCH == 6 || _M_ARM == 6 || defined(__ARM_ARCH_6__)
        PRINT *, 'INFO:arch[armv6]'
#elif __ARM_ARCH == 5 || _M_ARM == 5 || defined(__ARM_ARCH_5__)
        PRINT *, 'INFO:arch[armv5]'
#elif defined(__alpha) || defined(__alpha) || defined(_M_ALPHA)
        PRINT *, 'INFO:arch[alpha]'
#elif defined(__x86_64) || defined(__x86_64__) || defined(__amd64) ||         \
  defined(__amd64__) || defined(_M_X64) || defined(_M_AMD64)
        PRINT *, 'INFO:arch[x86_64]'
#elif defined(__i686) || defined(__i686__) || _M_IX86 == 600
        PRINT *, 'INFO:arch[i686]'
#elif defined(__i586) || defined(__i586__) || _M_IX86 == 500
        PRINT *, 'INFO:arch[i586]'
#elif defined(__i486) || defined(__i486__) || _M_IX86 == 400
        PRINT *, 'INFO:arch[i486]'
#elif defined(__i386) || defined(__i386__) || defined(_M_IX86)
        PRINT *, 'INFO:arch[i386]'
#elif defined(__ia64) || defined(__ia64__) || defined(_M_IA64)
        PRINT *, 'INFO:arch[ia64]'
#elif defined(__loongarch64)
        PRINT *, 'INFO:arch[loongarch64]'
#elif defined(__loongarch__)
        PRINT *, 'INFO:arch[loongarch32]'
#elif defined(__m68k__)
        PRINT *, 'INFO:arch[m68k]'
#elif defined(__mips64) || defined(__mips64__)
#  if defined(_MIPSEL)
        PRINT *, 'INFO:arch[mips64el]'
#  else
        PRINT *, 'INFO:arch[mips64]'
#  endif
#elif defined(__mips) || defined(__mips__)
#  if defined(_MIPSEL)
        PRINT *, 'INFO:arch[mipsel]'
#  else
        PRINT *, 'INFO:arch[mips]'
#  endif
#elif defined(__riscv) && __riscv_xlen == 64
        PRINT *, 'INFO:arch[riscv64]'
#elif defined(__riscv) && __riscv_xlen == 32
        PRINT *, 'INFO:arch[riscv32]'
#elif defined(__sw_64) || defined(__sw_64__)
        PRINT *, 'INFO:arch[sw_64]'
#elif defined(__s390x__)
        PRINT *, 'INFO:arch[s390x]'
#elif defined(__s390__)
        PRINT *, 'INFO:arch[s390]'
#elif defined(__sparcv9) || defined(__sparcv9__) || defined(__sparc64__)
        PRINT *, 'INFO:arch[sparcv9]'
#elif defined(__sparc) || defined(__sparc__)
        PRINT *, 'INFO:arch[sparc]'
#elif defined(__hppa) || defined(__hppa__)
#  if defined(__LP64__)
        PRINT *, 'INFO:arch[parisc64]'
#  else
        PRINT *, 'INFO:arch[parisc]'
#  endif
#elif defined(__ppc64__) || defined(__powerpc64__) || defined(__PPC64__) ||   \
  defined(_ARCH_PPC64)
#  if defined(_LITTLE_ENDIAN) || defined(__LITTLE_ENDIAN__)
        PRINT *, 'INFO:arch[ppc64le]'
#  else
        PRINT *, 'INFO:arch[ppc64]'
#  endif
#elif defined(__ppc__) || defined(__powerpc__) || defined(__PPC__) ||         \
  defined(_ARCH_PPC)
#  if defined(_LITTLE_ENDIAN) || defined(__LITTLE_ENDIAN__)
        PRINT *, 'INFO:arch[ppcle]'
#  else
        PRINT *, 'INFO:arch[ppc]'
#  endif
#endif

        PRINT *, 'ABI Detection'
      END


export void ispcCompilerABI() {

#if defined(__GNU__) && defined(__ELF__) && defined(__ARM_EABI__)
  print("INFO:abi[ELF ARMEABI]");
  static char const info_abi[] =
#elif defined(__GNU__) && defined(__ELF__) && defined(__ARMEB__)
  print("INFO:abi[ELF ARM]");
#elif defined(__GNU__) && defined(__ELF__) && defined(__ARMEL__)
  print("INFO:abi[ELF ARM]");

#elif defined(__linux__) && defined(__ELF__) && defined(__amd64__) &&         \
  defined(__ILP32__)
print("INFO:abi[ELF X32]");

#elif defined(__ELF__)
print("INFO:abi[ELF]");
#endif

}

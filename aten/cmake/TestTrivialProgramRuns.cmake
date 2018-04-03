# Check that our programs run.  This is different from the native CMake compiler
# check, which just tests if the program compiles and links.  This is important
# because with ASAN you might need to help the compiled library find some
# dynamic libraries.
CHECK_C_SOURCE_RUNS("
int main() { return 0; }
" COMPILER_WORKS)
IF(NOT COMPILER_WORKS)
  # Force cmake to retest next time around
  unset(COMPILER_WORKS CACHE)
  MESSAGE(FATAL_ERROR
      "Could not run a simple program built with your compiler. "
      "If you are trying to use -fsanitize=address, make sure "
      "libasan is properly installed on your system (you can confirm "
      "if the problem is this by attempting to build and run a "
      "small program.)")
ENDIF()

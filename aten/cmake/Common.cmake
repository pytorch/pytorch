# Mandatory-ish cmake settings

# if() recognizes numbers and boolean constants
cmake_policy(SET CMP0012 NEW)

if(UNIX)
  # prevent Unknown CMake command "check_function_exists".
  include(CheckFunctionExists)
endif()
include(CheckIncludeFile)
include(CheckCSourceCompiles)
include(CheckCSourceRuns)

include(CompilerSettings)
include(CompilerRPATH)
include(CheckCXXSourceCompiles)
include(TestForGlibCXXUseC99) # Test for an GCC 5 bug on Ubuntu 17.10 and newer
include(TestTrivialProgramRuns)
include(TestForCpuid)
include(TestForGccEbxFpicBug)
include(TestForAtomics)

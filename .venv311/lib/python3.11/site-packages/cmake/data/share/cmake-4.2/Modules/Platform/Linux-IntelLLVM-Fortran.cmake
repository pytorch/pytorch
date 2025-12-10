include(Platform/Linux-IntelLLVM)
__linux_compiler_intel_llvm(Fortran)
string(APPEND CMAKE_SHARED_LIBRARY_CREATE_Fortran_FLAGS " -nofor-main")
set(CMAKE_SHARED_LIBRARY_LINK_Fortran_FLAGS "")

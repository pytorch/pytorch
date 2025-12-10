include(Platform/Windows-MSVC)
if(NOT CMAKE_C_COMPILER_VERSION VERSION_LESS 18.0)
  set(_FS_C " /FS")
endif()
__windows_compiler_msvc(C)

if((NOT DEFINED CMAKE_DEPENDS_USE_COMPILER OR CMAKE_DEPENDS_USE_COMPILER)
    AND CMAKE_GENERATOR MATCHES "Makefiles|WMake"
    AND CMAKE_DEPFILE_FLAGS_C)
  # dependencies are computed by the compiler itself
  set(CMAKE_C_DEPENDS_USE_COMPILER TRUE)
endif()

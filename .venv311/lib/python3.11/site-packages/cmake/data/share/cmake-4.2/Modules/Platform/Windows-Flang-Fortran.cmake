include(Platform/Windows-MSVC)
__windows_compiler_msvc(Fortran)
set(CMAKE_Fortran_COMPILE_OBJECT "<CMAKE_Fortran_COMPILER> ${_COMPILE_Fortran} <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")

set(CMAKE_Fortran_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreaded         -Xclang --dependent-lib=libcmt)
set(CMAKE_Fortran_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDLL      -Xclang --dependent-lib=msvcrt)
set(CMAKE_Fortran_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebug    -Xclang --dependent-lib=libcmtd)
set(CMAKE_Fortran_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebugDLL -Xclang --dependent-lib=msvcrtd)

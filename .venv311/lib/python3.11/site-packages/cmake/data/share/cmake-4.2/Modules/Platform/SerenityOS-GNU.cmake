# This module is shared by multiple languages; use include blocker.
include_guard()

set(CMAKE_EXE_LINKER_FLAGS_INIT "-Wl,--hash-style=gnu,-z,relro,-z,now,-z,noexecstack,-z,separate-code,-z,max-page-size=0x1000")

macro(__serenity_compiler_gnu lang)
  set(CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG "-Wl,-rpath,")
  set(CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG_SEP ":")
  set(CMAKE_SHARED_LIBRARY_RPATH_LINK_${lang}_FLAG "-Wl,-rpath-link,")
  set(CMAKE_SHARED_LIBRARY_SONAME_${lang}_FLAG "-Wl,-soname,")
  set(CMAKE_EXE_EXPORTS_${lang}_FLAG "-Wl,--export-dynamic")

  set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS "-shared -Wl,--hash-style=gnu,-z,relro,-z,now,-z,noexecstack,-z,separate-code")

  # Initialize link type selection flags.  These flags are used when
  # building a shared library, shared module, or executable that links
  # to other libraries to select whether to use the static or shared
  # versions of the libraries.
  foreach(type SHARED_LIBRARY SHARED_MODULE EXE)
    set(CMAKE_${type}_LINK_STATIC_${lang}_FLAGS "-Wl,-Bstatic")
    set(CMAKE_${type}_LINK_DYNAMIC_${lang}_FLAGS "-Wl,-Bdynamic")
  endforeach()

endmacro()

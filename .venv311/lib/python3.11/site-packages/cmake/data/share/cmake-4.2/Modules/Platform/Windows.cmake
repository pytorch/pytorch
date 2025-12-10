set(CMAKE_STATIC_LIBRARY_PREFIX "")
set(CMAKE_STATIC_LIBRARY_SUFFIX ".lib")
set(CMAKE_SHARED_LIBRARY_PREFIX "")          # lib
if(CMAKE_SYSTEM_NAME STREQUAL "WindowsKernelModeDriver")
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".sys")          # .so
else()
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".dll")          # .so
endif()
set(CMAKE_IMPORT_LIBRARY_PREFIX "")
set(CMAKE_IMPORT_LIBRARY_SUFFIX ".lib")
set(CMAKE_EXECUTABLE_SUFFIX ".exe")          # .exe
set(CMAKE_LINK_LIBRARY_SUFFIX ".lib")
set(CMAKE_DL_LIBS "")
set(CMAKE_EXTRA_LINK_EXTENSIONS ".targets")

set(CMAKE_FIND_LIBRARY_PREFIXES
  "" # static or import library from MSVC tooling
  "lib" # static library from Meson with MSVC tooling
  )
set(CMAKE_FIND_LIBRARY_SUFFIXES
  ".dll.lib" # import library from Rust toolchain for MSVC ABI
  ".lib" # static or import library from MSVC tooling
  ".a" # static library from Meson with MSVC tooling
  )

# for borland make long command lines are redirected to a file
# with the following syntax, see Windows-bcc32.cmake for use
if(CMAKE_GENERATOR MATCHES "Borland")
  set(CMAKE_START_TEMP_FILE "@&&|\n")
  set(CMAKE_END_TEMP_FILE "\n|")
endif()

# for nmake make long command lines are redirected to a file
# with the following syntax, see Windows-bcc32.cmake for use
if(CMAKE_GENERATOR MATCHES "NMake")
  set(CMAKE_START_TEMP_FILE "@<<\n")
  set(CMAKE_END_TEMP_FILE "\n<<")
endif()

include(Platform/WindowsPaths)

# uncomment these out to debug nmake and borland makefiles
#set(CMAKE_START_TEMP_FILE "")
#set(CMAKE_END_TEMP_FILE "")
#set(CMAKE_VERBOSE_MAKEFILE 1)

# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__WINDOWS_EMBARCADERO)
  return()
endif()
set(__WINDOWS_EMBARCADERO 1)

set(BORLAND 1)

set(__pch_header_C "c-header")
set(__pch_header_CXX "c++-header")
set(__pch_header_OBJC "objective-c-header")
set(__pch_header_OBJCXX "objective-c++-header")

if("${CMAKE_${_lang}_COMPILER_VERSION}" VERSION_LESS 6.30)
  # Borland target type flags (bcc32 -h -t):
  set(_tW "-tW")       # -tW  GUI App         (implies -U__CONSOLE__)
  set(_tC "-tWC")      # -tWC Console App     (implies -D__CONSOLE__=1)
  set(_tD "-tWD")      # -tWD Build a DLL     (implies -D__DLL__=1 -D_DLL=1)
  set(_tM "-tWM")      # -tWM Enable threads  (implies -D__MT__=1 -D_MT=1)
  set(_tR "-tWR -tW-") # -tWR Use DLL runtime (implies -D_RTLDLL, and '-tW' too!!)
  # Notes:
  #  - The flags affect linking so we pass them to the linker.
  #  - The flags affect preprocessing so we pass them to the compiler.
  #  - Since '-tWR' implies '-tW' we use '-tWR -tW-' instead.
  #  - Since '-tW-' disables '-tWD' we use '-tWR -tW- -tWD' for DLLs.
else()
  set(EMBARCADERO 1)
  set(_tC "-tC") # Target is a console application
  set(_tD "-tD") # Target is a shared library
  set(_tM "-tM") # Target is multi-threaded
  set(_tR "-tR") # Target uses the dynamic RTL
  set(_tW "-tW") # Target is a Windows application
endif()
set(_COMPILE_C "")
set(_COMPILE_CXX " -P")

set(CMAKE_LIBRARY_PATH_FLAG "-L")
set(CMAKE_LINK_LIBRARY_FLAG "")

set(CMAKE_FIND_LIBRARY_SUFFIXES "-bcc.lib" ".lib")

# uncomment these out to debug makefiles
#set(CMAKE_START_TEMP_FILE "")
#set(CMAKE_END_TEMP_FILE "")
#set(CMAKE_VERBOSE_MAKEFILE 1)

# Borland cannot handle + in the file name, so mangle object file name
set (CMAKE_MANGLE_OBJECT_FILE_NAMES "ON")

set (CMAKE_BUILD_TYPE Debug CACHE STRING
     "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel.")

foreach(t EXE SHARED MODULE)
  string(APPEND CMAKE_${t}_LINKER_FLAGS_INIT " ${_tM} -lS:1048576 -lSc:4098 -lH:1048576 -lHc:8192 ")
  string(APPEND CMAKE_${t}_LINKER_FLAGS_DEBUG_INIT " -v")
  string(APPEND CMAKE_${t}_LINKER_FLAGS_RELWITHDEBINFO_INIT " -v")
endforeach()

# The Borland link tool does not support multiple concurrent
# invocations within a single working directory.
if(NOT DEFINED CMAKE_JOB_POOL_LINK)
  set(CMAKE_JOB_POOL_LINK BCC32LinkPool)
  get_property(_bccjp GLOBAL PROPERTY JOB_POOLS)
  if(NOT _bccjp MATCHES "BCC32LinkPool=")
      set_property(GLOBAL APPEND PROPERTY JOB_POOLS BCC32LinkPool=1)
  endif()
  unset(_bccjp)
endif()

macro(__embarcadero_language lang)
  set(CMAKE_${lang}_COMPILE_OPTIONS_DLL "${_tD}") # Note: This variable is a ';' separated list
  set(CMAKE_SHARED_LIBRARY_${lang}_FLAGS "${_tD}") # ... while this is a space separated string.
  set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS "")
  set(CMAKE_SHARED_MODULE_CREATE_${lang}_FLAGS "")
  set(CMAKE_${lang}_USE_RESPONSE_FILE_FOR_INCLUDES 1)

  set (CMAKE_${lang}_LINKER_WRAPPER_FLAG "-l")

  # compile a source file into an object file
  # place <DEFINES> outside the response file because Borland refuses
  # to parse quotes from the response file.
  set(CMAKE_${lang}_COMPILE_OBJECT
    "<CMAKE_${lang}_COMPILER> ${_tR} -DWIN32 <DEFINES> <INCLUDES> <FLAGS> -o<OBJECT>${_COMPILE_${lang}} -c <SOURCE>"
    )

  set(CMAKE_${lang}_LINK_EXECUTABLE
    "<CMAKE_${lang}_COMPILER> ${_tR} -e<TARGET> <LINK_FLAGS> <FLAGS> ${CMAKE_START_TEMP_FILE} <LINK_LIBRARIES> <OBJECTS>${CMAKE_END_TEMP_FILE}"
    # "implib -c -w <TARGET_IMPLIB> <TARGET>"
    )

  # place <DEFINES> outside the response file because Borland refuses
  # to parse quotes from the response file.
  set(CMAKE_${lang}_CREATE_PREPROCESSED_SOURCE
    "cpp32 -DWIN32 <DEFINES> <INCLUDES> <FLAGS> -o<PREPROCESSED_SOURCE>${_COMPILE_${lang}} -c <SOURCE>"
    )
  # Borland >= 5.6 allows -P option for cpp32, <= 5.5 does not

  # Create a module library.
  set(CMAKE_${lang}_CREATE_SHARED_MODULE
    "<CMAKE_${lang}_COMPILER> ${_tR} ${_tD} ${CMAKE_START_TEMP_FILE}-e<TARGET> <LINK_FLAGS> <LINK_LIBRARIES> <OBJECTS>${CMAKE_END_TEMP_FILE}"
    )

  # Create an import library for another target.
  set(CMAKE_${lang}_CREATE_IMPORT_LIBRARY
    "implib -c -w <TARGET_IMPLIB> <TARGET>"
    )

  # Create a shared library.
  # First create a module and then its import library.
  set(CMAKE_${lang}_CREATE_SHARED_LIBRARY
    ${CMAKE_${lang}_CREATE_SHARED_MODULE}
    ${CMAKE_${lang}_CREATE_IMPORT_LIBRARY}
    )

  # create a static library
  set(CMAKE_${lang}_CREATE_STATIC_LIBRARY
    "tlib ${CMAKE_START_TEMP_FILE}/p512 <LINK_FLAGS> /a <TARGET_QUOTED> <OBJECTS>${CMAKE_END_TEMP_FILE}"
    )

  set(CMAKE_${lang}_CREATE_WIN32_EXE "${_tW}")
  set(CMAKE_${lang}_CREATE_CONSOLE_EXE "${_tC}")

  # Precompile Headers
  if (EMBARCADERO)
    set(CMAKE_PCH_EXTENSION .pch)
    set(CMAKE_${lang}_COMPILE_OPTIONS_USE_PCH -Xclang -include-pch -Xclang <PCH_FILE> -Xclang -include -Xclang <PCH_HEADER>)
    set(CMAKE_${lang}_COMPILE_OPTIONS_CREATE_PCH -Xclang -emit-pch -Xclang -include -Xclang <PCH_HEADER> -x ${__pch_header_${lang}})
  endif()

  # Initial configuration flags.
  string(APPEND CMAKE_${lang}_FLAGS_INIT " ${_tM}")
  string(APPEND CMAKE_${lang}_FLAGS_DEBUG_INIT " -Od -v")
  string(APPEND CMAKE_${lang}_FLAGS_MINSIZEREL_INIT " -O1 -DNDEBUG")
  string(APPEND CMAKE_${lang}_FLAGS_RELEASE_INIT " -O2 -DNDEBUG")
  string(APPEND CMAKE_${lang}_FLAGS_RELWITHDEBINFO_INIT " -Od")
  set(CMAKE_${lang}_STANDARD_LIBRARIES_INIT "import32.lib")
endmacro()

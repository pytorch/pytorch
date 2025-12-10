# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(FortranCInterface C Fortran)
include(${FortranCInterface_BINARY_DIR}/Input.cmake OPTIONAL)

# Check if the C compiler supports '$' in identifiers.
include(CheckSourceCompiles)
check_source_compiles(C
"extern int dollar$(void);
int main() { return 0; }"
C_SUPPORTS_DOLLAR)

# List manglings of global symbol names to try.
set(global_symbols
  my_sub    # VisualAge
  my_sub_   # GNU, Intel, HP, SunPro, PGI
  my_sub__  # GNU g77
  MY_SUB    # Intel on Windows
  mysub     # VisualAge
  mysub_    # GNU, Intel, HP, SunPro, PGI
  MYSUB     # Intel on Windows
  ${FortranCInterface_GLOBAL_SYMBOLS}
  )
list(REMOVE_DUPLICATES global_symbols)

# List manglings of module symbol names to try.
set(module_symbols
  __my_module_MOD_my_sub  # GNU 4.3
  __my_module_NMOD_my_sub # VisualAge
  __my_module__my_sub     # GNU 4.2
  __mymodule_MOD_mysub    # GNU 4.3
  __mymodule_NMOD_mysub   # VisualAge
  __mymodule__mysub       # GNU 4.2
  my_module$my_sub        # HP
  my_module_mp_my_sub_    # Intel
  MY_MODULE_mp_MY_SUB     # Intel on Windows
  my_module_my_sub_       # PGI
  my_module_MP_my_sub     # NAG
  mymodule$mysub          # HP
  mysub$mymodule_         # Cray
  my_sub$my_module_       # Cray
  mymodule_mp_mysub_      # Intel
  MYMODULE_mp_MYSUB       # Intel on Windows
  mymodule_mysub_         # PGI
  mymodule_MP_mysub       # NAG
  _QMmy_modulePmy_sub     # LLVMFlang
  _QMmymodulePmysub       # LLVMFlang
  __module_my_module_my_sub # LFortran
  __module_mymodule_mysub   # LFortran
  ${FortranCInterface_MODULE_SYMBOLS}
  )
list(REMOVE_DUPLICATES module_symbols)

# Note that some compiler manglings cannot be invoked from C:
#   SunPro uses "my_module.my_sub_"
#   PathScale uses "MY_SUB.in.MY_MODULE"

# Add module symbols only with Fortran90.
if(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
  set(myfort_modules mymodule.f90 my_module.f90)
  set(call_mod call_mod.f90)
  set_property(SOURCE main.F PROPERTY COMPILE_DEFINITIONS CALL_MOD)
else()
  set(module_symbols)
endif()

# Generate C symbol sources.
set(symbol_sources)
if(NOT CMAKE_Fortran_COMPILER_ID MATCHES "^(PathScale)$")
  # Provide mymodule_ and my_module_ init symbols because:
  #  - PGI Fortran uses module init symbols
  #  - Cray Fortran >= 7.3.2 uses module init symbols
  # but not for:
  #  - PathScale Fortran uses module init symbols but module symbols
  #    use '.in.' so we cannot provide them anyway.
  list(APPEND symbol_sources mymodule_.c my_module_.c MY_MODULE.c MYMODULE.c)
endif()
foreach(symbol IN LISTS global_symbols module_symbols)
  # Skip symbols with '$' if C cannot handle them.
  if(C_SUPPORTS_DOLLAR OR NOT "${symbol}" MATCHES "\\$")
    if("${symbol}" MATCHES "SUB")
      set(upper "-UPPER")
    else()
      set(upper)
    endif()
    string(REPLACE "$" "S" name "${symbol}")
    set(source ${CMAKE_CURRENT_BINARY_DIR}/symbols/${name}${upper}.c)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/symbol.c.in ${source} @ONLY)
    list(APPEND symbol_sources ${source})
  endif()
endforeach()

# Provide symbols through Fortran.
add_library(myfort STATIC mysub.f my_sub.f ${myfort_modules})

# Provide symbols through C but fall back to Fortran.
add_library(symbols STATIC ${symbol_sources})
target_link_libraries(symbols PUBLIC myfort)

# In case the Fortran compiler produces PIC by default make sure
# the C compiler produces PIC even if it is not its default.
set_property(TARGET symbols PROPERTY POSITION_INDEPENDENT_CODE 1)

if(CMAKE_Fortran_COMPILER_ID STREQUAL "LFortran")
  if(CMAKE_Fortran_COMPILER_VERSION VERSION_GREATER_EQUAL "0.55")
    set(_LFORTRAN_SC_FLAG --separate-compilation)
  else()
    set(_LFORTRAN_SC_FLAG --generate-object-code)
  endif()
  add_compile_options(--implicit-interface ${_LFORTRAN_SC_FLAG})
endif()

# Require symbols through Fortran.
add_executable(FortranCInterface main.F call_sub.f ${call_mod})
target_link_libraries(FortranCInterface PUBLIC symbols)

# If IPO is enabled here, GCC gfortran >= 12.0 will obfuscate
# the strings of the return values in the compiled executable,
# which we use to regex match against later.
# The static libraries must be build with IPO and non-IPO objects,
# as that will ensure the verify step will operate on IPO objects,
# if requested by the system compiler flags.
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU" AND
  CMAKE_Fortran_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
  target_compile_options(FortranCInterface PRIVATE "-fno-lto")
  if(NOT APPLE)
    target_compile_options(myfort PRIVATE "-flto=auto" "-ffat-lto-objects")
  endif()
endif()
if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND
  CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
  if(NOT APPLE)
    target_compile_options(symbols PRIVATE "-flto=auto" "-ffat-lto-objects")
  endif()
endif()

file(GENERATE OUTPUT exe-$<CONFIG>.cmake CONTENT [[
set(FortranCInterface_EXE "$<TARGET_FILE:FortranCInterface>")
]])

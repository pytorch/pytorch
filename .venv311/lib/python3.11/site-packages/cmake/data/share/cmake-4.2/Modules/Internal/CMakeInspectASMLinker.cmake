# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# Re-configure to save learned information.
block()
  foreach(_var IN ITEMS
      # Compiler information.
      # Keep in sync with CMakeDetermineASMCompiler.
      COMPILER
      COMPILER_ID
      COMPILER_ARG1
      COMPILER_ENV_VAR
      COMPILER_AR
      COMPILER_RANLIB
      COMPILER_VERSION
      COMPILER_ARCHITECTURE_ID
      # Linker information.
      COMPILER_LINKER
      COMPILER_LINKER_ID
      COMPILER_LINKER_VERSION
      COMPILER_LINKER_FRONTEND_VARIANT
      LINKER_DEPFILE_SUPPORTED
      LINKER_PUSHPOP_STATE_SUPPORTED
      )
    set(_CMAKE_ASM_${_var} "${CMAKE_ASM${ASM_DIALECT}_${_var}}")
  endforeach()
  configure_file(
    ${CMAKE_ROOT}/Modules/CMakeASMCompiler.cmake.in
    ${CMAKE_PLATFORM_INFO_DIR}/CMakeASM${ASM_DIALECT}Compiler.cmake
    @ONLY)
endblock()

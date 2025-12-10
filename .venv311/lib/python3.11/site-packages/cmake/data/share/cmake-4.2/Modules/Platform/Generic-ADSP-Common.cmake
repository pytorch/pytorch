# support for the Analog Devices toolchain for their DSPs
# Raphael Cotty" <raphael.cotty (AT) googlemail.com>
#
# it supports three architectures:
# Blackfin
# TS (TigerShark)
# 21k (Sharc 21xxx)

if(NOT ADSP)

  set(ADSP TRUE)

  set(CMAKE_STATIC_LIBRARY_SUFFIX ".dlb")
  set(CMAKE_SHARED_LIBRARY_SUFFIX "")
  set(CMAKE_EXECUTABLE_SUFFIX ".dxe")

  # if ADSP_PROCESSOR has not been set, but CMAKE_SYSTEM_PROCESSOR has,
  # assume that this is the processor name to use for the compiler
  if(CMAKE_SYSTEM_PROCESSOR AND NOT ADSP_PROCESSOR)
    set(ADSP_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR})
  endif()

  # if ADSP_PROCESSOR_SILICIUM_REVISION has not been set, use "none"
  if(NOT ADSP_PROCESSOR_SILICIUM_REVISION)
    set(ADSP_PROCESSOR_SILICIUM_REVISION "none")
  endif()

  # this file is included from the C and CXX files, so handle both here

  get_filename_component(_ADSP_DIR "${CMAKE_C_COMPILER}" PATH)
  if(NOT _ADSP_DIR)
    get_filename_component(_ADSP_DIR "${CMAKE_CXX_COMPILER}" PATH)
  endif()
  if(NOT _ADSP_DIR)
    get_filename_component(_ADSP_DIR "${CMAKE_ASM_COMPILER}" PATH)
  endif()

  # detect architecture

  if(CMAKE_C_COMPILER MATCHES ccblkfn OR CMAKE_CXX_COMPILER MATCHES ccblkfn OR CMAKE_ASM_COMPILER MATCHES easmBLKFN)
    if(NOT ADSP_PROCESSOR)
      set(ADSP_PROCESSOR "ADSP-BF561")
    endif()
    set(ADSP_BLACKFIN TRUE)
    set(_ADSP_FAMILY_DIR "${_ADSP_DIR}/Blackfin")
  endif()

  if(CMAKE_C_COMPILER MATCHES ccts OR CMAKE_CXX_COMPILER MATCHES ccts OR CMAKE_ASM_COMPILER MATCHES easmTS)
    if(NOT ADSP_PROCESSOR)
      set(ADSP_PROCESSOR "ADSP-TS101")
    endif()
    set(ADSP_TS TRUE)
    set(_ADSP_FAMILY_DIR "${_ADSP_DIR}/TS")
  endif()

  if(CMAKE_C_COMPILER MATCHES cc21k OR CMAKE_CXX_COMPILER MATCHES cc21k OR CMAKE_ASM_COMPILER MATCHES easm21k)
    if(NOT ADSP_PROCESSOR)
      set(ADSP_PROCESSOR "ADSP-21060")
    endif()
    set(ADSP_21K TRUE)

    set(_ADSP_FAMILY_DIR "${_ADSP_DIR}/21k")  # default if nothing matches
    if   (ADSP_PROCESSOR MATCHES "210..$")
      set(_ADSP_FAMILY_DIR "${_ADSP_DIR}/21k")
    endif()

    if   (ADSP_PROCESSOR MATCHES "211..$")
      set(_ADSP_FAMILY_DIR "${_ADSP_DIR}/211k")
    endif()

    if   (ADSP_PROCESSOR MATCHES "212..$")
      set(_ADSP_FAMILY_DIR "${_ADSP_DIR}/212k")
    endif()

    if   (ADSP_PROCESSOR MATCHES "213..$")
      set(_ADSP_FAMILY_DIR "${_ADSP_DIR}/213k")
    endif()

    set(_ADSP_FAMILY_DIR "${_ADSP_DIR}/21k")
  endif()


  link_directories("${_ADSP_FAMILY_DIR}/lib")

  # vdk support
  find_program( ADSP_VDKGEN_EXECUTABLE vdkgen "${_ADSP_FAMILY_DIR}/vdk" )

  macro(ADSP_GENERATE_VDK VDK_GENERATED_HEADER VDK_GENERATED_SOURCE VDK_KERNEL_SUPPORT_FILE)
    add_custom_command(
      OUTPUT ${VDK_GENERATED_HEADER} ${VDK_GENERATED_SOURCE}
      COMMAND ${ADSP_VDKGEN_EXECUTABLE} ${VDK_KERNEL_SUPPORT_FILE} -proc ${ADSP_PROCESSOR} -si-revision ${ADSP_PROCESSOR_SILICIUM_REVISION} -MM
      DEPENDS ${VDK_KERNEL_SUPPORT_FILE}
      )
  endmacro()

  # loader support
  find_program( ADSP_ELFLOADER_EXECUTABLE elfloader "${_ADSP_FAMILY_DIR}" )

  # BOOT_MODE: prom, flash, spi, spislave, UART, TWI, FIFO
  # FORMAT: hex, ASCII, binary, include
  # WIDTH: 8, 16
  macro(ADSP_CREATE_LOADER_FILE TARGET_NAME BOOT_MODE FORMAT WIDTH)
    add_custom_command(
      TARGET ${TARGET_NAME}
      POST_BUILD
      COMMAND ${ADSP_ELFLOADER_EXECUTABLE} ${EXECUTABLE_OUTPUT_PATH}/${TARGET_NAME}.dxe -proc ${ADSP_PROCESSOR} -si-revision ${ADSP_PROCESSOR_SILICIUM_REVISION} -b ${BOOT_MODE} -f ${FORMAT} -width ${WIDTH} -o ${EXECUTABLE_OUTPUT_PATH}/${TARGET_NAME}.ldr
      COMMENT "Building the loader file"
      )
  endmacro()

  macro(ADSP_CREATE_LOADER_FILE_INIT TARGET_NAME BOOT_MODE FORMAT WIDTH INITIALIZATION_FILE)
    add_custom_command(
      TARGET ${TARGET_NAME}
      POST_BUILD
      COMMAND ${ADSP_ELFLOADER_EXECUTABLE} ${EXECUTABLE_OUTPUT_PATH}/${TARGET_NAME}.dxe -proc ${ADSP_PROCESSOR} -si-revision ${ADSP_PROCESSOR_SILICIUM_REVISION} -b ${BOOT_MODE} -f ${FORMAT} -width ${WIDTH} -o ${EXECUTABLE_OUTPUT_PATH}/${TARGET_NAME}.ldr -init ${INITIALIZATION_FILE}
      COMMENT "Building the loader file"
      )
  endmacro()

endif()

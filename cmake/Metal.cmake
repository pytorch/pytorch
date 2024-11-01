if(NOT APPLE)
    return()
endif()

set(METAL_CFLAGS -Wall -Wextra -fno-fast-math)
if(WERROR)
    string(APPEND METAL_CFLAGS -Werror)
endif()

function(metal_to_air SRC TARGET FLAGS)
    add_custom_command(COMMAND xcrun metal -c ${SRC} -o ${TARGET} ${FLAGS} ${METAL_CFLAGS}
                       DEPENDS ${SRC}
                       OUTPUT ${TARGET}
                       COMMENT "Compiling ${SRC} to ${TARGET}"
                       VERBATIM)
endfunction()

function(air_to_metallib TARGET OBJECTS)
    set(_OBJECTS ${OBJECTS} ${ARGN})
    add_custom_command(COMMAND xcrun metallib -o ${TARGET} ${_OBJECTS}
                       DEPENDS ${_OBJECTS}
                       OUTPUT ${TARGET}
                       COMMENT "Linking ${TARGET}"
                       VERBATIM)
endfunction()

function(metal_to_metallib_h SRC TGT)
    file(READ ${SRC} SHADER_CONTENT)
    file(WRITE ${TGT} "#include <ATen/native/mps/OperationUtils.h>\n")
    file(APPEND ${TGT} "static ::at::native::mps::MetalShaderLibrary lib(R\"SHDR(\n")
    file(APPEND ${TGT} "${SHADER_CONTENT}")
    file(APPEND ${TGT} ")SHDR\");\n")
endfunction()

set(BFLOAT_METAL_CODE "
  kernel void inc(device bfloat* ptr,
                   uint idx [[thread_position_in_grid]]) {
    ptr[idx] += 1;
  }
")
if(NOT CAN_COMPILE_METAL_FOUND)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/bfloat_inc.metal" "${BFLOAT_METAL_CODE}")
    execute_process(COMMAND xcrun metal -std=metal3.1 bfloat_inc.metal
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                    OUTPUT_VARIABLE XCRUN_OUTPUT
                    ERROR_VARIABLE XCRUN_OUTPUT
                    RESULT_VARIABLE XCRUN_RC)
    if(${XCRUN_RC} EQUAL 0)
        message(STATUS "Machine can compile metal shaders")
        set(CAN_COMPILE_METAL YES CACHE BOOL "Host can compile metal shaders")
    else()
        message(WARNING "Machine can not compile metal shaders, fails with ${XCRUN_OUTPUT}")
        set(CAN_COMPILE_METAL NO CACHE BOOL "Host can compile metal shaders")
    endif()
    set(CAN_COMPILE_METAL_FOUND YES CACHE INTERNAL "Run check for shader compiler")
endif()

if(NOT USE_PYTORCH_METAL)
    return()
endif()

if(IOS OR INTERN_BUILD_MOBILE)
    return()
endif()

set(OSX_PLATFORM "MacOSX.platform")
exec_program(/usr/bin/xcode-select ARGS -print-path OUTPUT_VARIABLE CMAKE_XCODE_DEVELOPER_DIR)
set(XCODE_POST_43_ROOT "${CMAKE_XCODE_DEVELOPER_DIR}/Platforms/${OSX_PLATFORM}/Developer")
set(XCODE_PRE_43_ROOT "/Developer/Platforms/${OSX_PLATFORM}/Developer")
if(NOT DEFINED CMAKE_OSX_DEVELOPER_ROOT)
    if(EXISTS ${XCODE_POST_43_ROOT})
        set(CMAKE_OSX_DEVELOPER_ROOT ${XCODE_POST_43_ROOT})
    elseif(EXISTS ${XCODE_PRE_43_ROOT})
        set(CMAKE_OSX_DEVELOPER_ROOT ${XCODE_PRE_43_ROOT})
    elseif(EXISTS ${CMAKE_XCODE_DEVELOPER_DIR} AND ${CMAKE_XCODE_DEVELOPER_DIR} STREQUAL "/Library/Developer/CommandLineTools")
            set(CMAKE_OSX_DEVELOPER_ROOT ${CMAKE_XCODE_DEVELOPER_DIR})
    endif()
endif(NOT DEFINED CMAKE_OSX_DEVELOPER_ROOT)
set(CMAKE_OSX_DEVELOPER_ROOT ${CMAKE_OSX_DEVELOPER_ROOT} CACHE PATH "Location of OSX SDKs root directory")

if(NOT DEFINED CMAKE_OSX_SDK_ROOT)
    file(GLOB _CMAKE_OSX_SDKS "${CMAKE_OSX_DEVELOPER_ROOT}/SDKs/*")
    if(_CMAKE_OSX_SDKS)
        list(SORT _CMAKE_OSX_SDKS)
        list(REVERSE _CMAKE_OSX_SDKS)
        list(GET _CMAKE_OSX_SDKS 0 CMAKE_OSX_SDK_ROOT)
        message(STATUS "_CMAKE_OSX_SDKS: ${_CMAKE_OSX_SDKS}")
    else(_CMAKE_OSX_SDKS)
        message(FATAL_ERROR "No OSX SDK's found in default search path ${CMAKE_OSX_DEVELOPER_ROOT}.")
    endif(_CMAKE_OSX_SDKS)
    message(STATUS "Toolchain using default OSX SDK: ${CMAKE_OSX_SDK_ROOT}")
endif(NOT DEFINED CMAKE_OSX_SDK_ROOT)
set(CMAKE_OSX_SDK_ROOT ${CMAKE_OSX_SDK_ROOT} CACHE PATH "Location of the selected OSX SDK")
set(CMAKE_FRAMEWORK_PATH
    ${CMAKE_OSX_SDK_ROOT}/System/Library/Frameworks
    ${CMAKE_OSX_SDK_ROOT}/System/Library/PrivateFrameworks
    ${CMAKE_OSX_SDK_ROOT}/Developer/Library/Frameworks
)
message(STATUS "CMAKE_FRAMEWORK_PATH: ${CMAKE_FRAMEWORK_PATH}")
set(CMAKE_FIND_FRAMEWORK FIRST)

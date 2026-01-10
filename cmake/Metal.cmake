if(NOT APPLE)
    return()
endif()

# macOS SDK version
execute_process(
    COMMAND zsh "-c" "/usr/bin/xcrun -sdk macosx --show-sdk-version"
    OUTPUT_VARIABLE MACOS_SDK_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)

# Metal version
execute_process(
    COMMAND
        zsh "-c"
        "echo \"__METAL_VERSION__\" | xcrun -sdk macosx metal ${XCRUN_FLAGS} -E -x metal -P - | tail -1 | tr -d '\n'"
    OUTPUT_VARIABLE MPS_METAL_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)

# Check for Metal 4 support based on Metal version
if(MPS_METAL_VERSION GREATER_EQUAL 400)
    set(MPS_METAL_4 TRUE CACHE BOOL "Enable metal 4 for MPS")
else()
    set(MPS_METAL_4 FALSE CACHE BOOL "Enable metal 4 for MPS")
endif()

set(METAL_CFLAGS -Wall -Wextra -fno-fast-math)

if(MPS_METAL_4)
    set(METAL_FLAGS ${METAL_FLAGS} -Wno-c++20-extensions -std=metal4.0)
    list(APPEND METAL_CFLAGS ${METAL_FLAGS})
endif()

if(WERROR)
    list(APPEND METAL_CFLAGS -Werror)
endif()

function(metal_to_air SRC TARGET FLAGS)
    add_custom_command(COMMAND xcrun metal -c ${SRC} -I ${CMAKE_SOURCE_DIR} -I ${CMAKE_SOURCE_DIR}/aten/src -o ${TARGET} ${FLAGS} ${METAL_CFLAGS}
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
    execute_process(COMMAND ${Python_EXECUTABLE} torch/utils/_cpp_embed_headers.py ${SRC}
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                    OUTPUT_VARIABLE SHADER_CONTENT
                    RESULT_VARIABLE _exitcode)
    if(NOT _exitcode EQUAL 0)
        message(FATAL_ERROR "Failed to preprocess Metal shader ${SRC}")
        return()
    endif()
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
    execute_process(COMMAND xcrun metal ${METAL_CFLAGS} bfloat_inc.metal
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

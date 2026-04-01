#
# Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

# This file will only be executed if VMA_BUILD_SAMPLE_SHADERS is set to ON

find_program(GLSL_VALIDATOR glslangValidator REQUIRED)

if(NOT GLSL_VALIDATOR)
    message(FATAL_ERROR "glslangValidator not found!")
endif()

set(SHADERS
    Shader.vert
    Shader.frag
    SparseBindingTest.comp
)

# Compile each shader using glslangValidator
foreach(SHADER ${SHADERS})
    get_filename_component(FILE_NAME ${SHADER} NAME)
    
    # Put the .spv files into the bin folder
    set(SPIRV_BIN ${CMAKE_CURRENT_BINARY_DIR}/${FILE_NAME}.spv)
    set(SPIRV ${PROJECT_SOURCE_DIR}/bin/${FILE_NAME}.spv)


    add_custom_command(
        OUTPUT ${SPIRV}
        # Use the same file name and append .spv to the compiled shader
        COMMAND ${GLSL_VALIDATOR} -V ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER} -o ${SPIRV}
        DEPENDS ${SHADER}
    )
    add_custom_command(
        OUTPUT ${SPIRV_BIN}
        # Use the same file name and append .spv to the compiled shader
        COMMAND ${GLSL_VALIDATOR} -V ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER} -o ${SPIRV_BIN}
        DEPENDS ${SHADER}
    )

    list(APPEND SPIRV_FILES ${SPIRV} ${SPIRV_BIN})
endforeach()

add_custom_target(VmaSampleShaders ALL DEPENDS ${SPIRV_FILES})

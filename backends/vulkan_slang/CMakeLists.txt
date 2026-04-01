cmake_minimum_required(VERSION 3.18)
project(torch_vulkan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Find dependencies ──────────────────────────────────────────────
find_package(Vulkan REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Find PyTorch
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_FIND_RESULT
)
if(NOT TORCH_FIND_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not find PyTorch. Install with: pip install torch")
endif()

list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)

# ── VMA (Vulkan Memory Allocator) ──────────────────────────────────
set(VMA_DIR "${CMAKE_SOURCE_DIR}/third_party/VulkanMemoryAllocator")
if(NOT EXISTS "${VMA_DIR}/include/vk_mem_alloc.h")
    message(FATAL_ERROR "VMA not found. Run: git submodule update --init")
endif()

# ── Slang shader compiler ────────────────────────────────────────
set(SLANG_DIR "${CMAKE_SOURCE_DIR}/third_party/slang")
if(NOT EXISTS "${SLANG_DIR}/slang.h")
    message(WARNING "Slang submodule not found at ${SLANG_DIR}. "
                    "Run: git submodule update --init --recursive")
endif()

# ── Compile shaders (Slang → SPIR-V → C++ header) ────────────────
set(SHADER_HEADER "${CMAKE_SOURCE_DIR}/csrc/generated/shaders.h")
find_program(SLANGC slangc HINTS ENV SLANGC
    "${SLANG_DIR}/build/Release/bin"
    "${SLANG_DIR}/build/bin")

if(SLANGC)
    file(GLOB_RECURSE SHADER_SOURCES "${CMAKE_SOURCE_DIR}/shaders/*.slang")
    add_custom_command(
        OUTPUT ${SHADER_HEADER}
        COMMAND ${CMAKE_COMMAND} -E env "SLANGC=${SLANGC}"
                ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/compile_shaders.py
        DEPENDS ${SHADER_SOURCES} ${CMAKE_SOURCE_DIR}/tools/compile_shaders.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Compiling Slang shaders to SPIR-V"
    )
    add_custom_target(compile_shaders DEPENDS ${SHADER_HEADER})
else()
    message(WARNING "slangc not found. Using pre-compiled shaders.h if available. "
                    "Set SLANGC env var or build Slang: "
                    "cd third_party/slang && cmake --preset release && cmake --build --preset release")
    add_custom_target(compile_shaders)
endif()

# ── Collect sources ────────────────────────────────────────────────
file(GLOB_RECURSE VULKAN_RUNTIME_SRC "csrc/vulkan/*.cpp")
file(GLOB_RECURSE OPS_SRC "csrc/ops/*.cpp")
file(GLOB_RECURSE BACKEND_SRC "csrc/backend/*.cpp")
file(GLOB_RECURSE AUTOCAST_SRC "csrc/autocast/*.cpp")
set(INIT_SRC "csrc/init.cpp")

set(ALL_SOURCES
    ${VULKAN_RUNTIME_SRC}
    ${OPS_SRC}
    ${BACKEND_SRC}
    ${AUTOCAST_SRC}
    ${INIT_SRC}
)

# ── Build shared library ──────────────────────────────────────────
# Python extension naming: _C.cpython-XXX-linux-gnu.so
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_library(_C SHARED ${ALL_SOURCES})
add_dependencies(_C compile_shaders)
set_target_properties(_C PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_EXT_SUFFIX}"
)

target_include_directories(_C PRIVATE
    ${CMAKE_SOURCE_DIR}/csrc
    ${CMAKE_SOURCE_DIR}/csrc/generated
    ${VMA_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(_C PRIVATE
    ${TORCH_LIBRARIES}
    Vulkan::Vulkan
    Python3::Python
)

target_compile_definitions(_C PRIVATE
    VMA_STATIC_VULKAN_FUNCTIONS=0
    VMA_DYNAMIC_VULKAN_FUNCTIONS=1
)

# On Linux, hide symbols by default
if(UNIX AND NOT APPLE)
    target_compile_options(_C PRIVATE -fvisibility=hidden)
endif()

# Install to python package
install(TARGETS _C DESTINATION torch_vulkan)

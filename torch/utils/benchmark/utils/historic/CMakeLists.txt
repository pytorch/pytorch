# Fallback for back-testing earlier versions of PyTorch.
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(timer_cpp_jit)
set(CMAKE_CXX_STANDARD 14)

message("name:            ${NAME}")
message("torch_root:      ${TORCH_ROOT}")
message("is_standalone:   ${IS_STANDALONE}")
message("link_libtorch:   ${LINK_LIBTORCH}")
message("valgrind_include ${VALGRIND_INCLUDE}")

if(${IS_STANDALONE})
    add_executable(${NAME} "${PROJECT_SOURCE_DIR}/${NAME}.cpp")
else()
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(PythonInterp REQUIRED)  # Without this PyBind11 will find the wrong version
    find_package(pybind11 CONFIG REQUIRED)
    add_compile_definitions(TORCH_EXTENSION_NAME=${NAME})
    pybind11_add_module(${NAME} "${PROJECT_SOURCE_DIR}/${NAME}.cpp")
endif()

# Wall time doesn't need this, but it doesn't hurt.
target_include_directories(${NAME} PRIVATE ${VALGRIND_INCLUDE})

if(${LINK_LIBTORCH})
    list(APPEND CMAKE_PREFIX_PATH ${TORCH_ROOT})
    find_package(Torch REQUIRED)
    find_package(Threads REQUIRED)

    target_include_directories(${NAME} PRIVATE ${TORCH_INCLUDE_DIRS})

    # Older versions of PyTorch struggle to link MKL.
    target_link_directories(${NAME} PUBLIC "${LIB_PATH}")
    target_link_libraries(
        ${NAME} PRIVATE
        -Wl,--gc-sections
        -Wl,--whole-archive
        ${TORCH_LIBRARIES}
        -Wl,--no-whole-archive
        Threads::Threads
    )
endif()

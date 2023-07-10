cmake_minimum_required(VERSION 3.15)
project(ck_app)
add_compile_options(-std=c++17)

find_package(composable_kernel 1.0.0 COMPONENTS device_operations)
find_package(hip REQUIRED PATHS /opt/rocm)
message(STATUS "Build with HIP ${hip_VERSION}")

# add all example subdir
file(GLOB dir_list LIST_DIRECTORIES true *)
FOREACH(subdir ${dir_list})
    IF(IS_DIRECTORY "${subdir}" AND (NOT "${subdir}" MATCHES "build"))
        add_subdirectory(${subdir})
    ENDIF()
ENDFOREACH()

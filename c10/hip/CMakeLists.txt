# Build file for the C10 HIP.  This directory doesn't actually contain
# any files; these files are copied over by the HIPIFY script in tools/amd_build

# ---[ Configure macro file.
set(C10_HIP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}) # used in cmake_macros.h.in
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/impl/hip_cmake_macros.h.in
    ${CMAKE_BINARY_DIR}/c10/hip/impl/hip_cmake_macros.h)

# NB: All previous cu files are renamed into cc files.  This isn't tested at the
# moment.
file(GLOB C10_HIP_SRCS
        *.cpp
        *.cc
        impl/*.cpp
        impl/*.cc
        )

# Mark the cc files as HIP files, so we call the compiler.  (They have to be
# suffixed with cc, because the hcc compiler won't accept them otherwise.)
file(GLOB __c10_hip_srcs_cpp *.cc impl/*.cc)
set_source_files_properties(${__c10_hip_srcs_cpp} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

file(GLOB_RECURSE C10_HIP_HEADERS *.h)

set(C10_HIP_INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_BINARY_DIR}))))

add_subdirectory(test)

# ---[ Installation
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")
install(FILES ${CMAKE_BINARY_DIR}/c10/hip/impl/hip_cmake_macros.h
  DESTINATION include/c10/hip/impl)

# Pass source and includes to parents
set(C10_HIP_SRCS ${C10_HIP_SRCS} PARENT_SCOPE)
set(C10_HIP_INCLUDE ${C10_HIP_INCLUDE} PARENT_SCOPE)

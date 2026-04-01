# Make sure that shared debug info doesn't intefere with caching
# See the sccache readme
if(
    MSVC
    AND (NOT DEFINED CMAKE_MSVC_DEBUG_INFORMATION_FORMAT)
    AND (
        CMAKE_C_COMPILER_LAUNCHER MATCHES "ccache"
        OR CMAKE_CXX_COMPILER_LAUNCHER MATCHES "ccache"
    )
)
    message(
        NOTICE
        "Setting embedded debug info for MSVC to work around (s)ccache's inability to cache shared debug info files"
    )
    cmake_minimum_required(VERSION 3.25)
    cmake_policy(GET CMP0141 cmp0141)
    if(NOT cmp0141 STREQUAL "NEW")
        message(WARNING "Need CMake policy 0141 enabled")
    endif()
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
        "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>"
    )
endif()

#
# glob_append(MY_VAR my_glob) will append the results of file(GLOB
# CONFIGURE_DEPENDS my_glob) to MY_VAR
#
# Any number of globs may be specified
#
function(glob_append dest)
    file(GLOB files CONFIGURE_DEPENDS ${ARGN})
    list(APPEND ${dest} ${files})
    set(${dest} ${${dest}} PARENT_SCOPE)
endfunction()

#
# Perform a recursive glob, and exclude any files appropriately according to
# the host system and build options
#
function(slang_glob_sources var dir)
    set(patterns
        "*.cpp"
        "*.h"
        "*.natvis"
        "*.natstepfilter"
        "*.natjmc"
    )
    if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        list(APPEND patterns "*.mm")
    endif()
    list(TRANSFORM patterns PREPEND "${dir}/")

    file(GLOB_RECURSE files CONFIGURE_DEPENDS ${patterns})

    if(NOT WIN32)
        list(FILTER files EXCLUDE REGEX "(^|/)windows/.*")
    endif()

    if(NOT UNIX)
        list(FILTER files EXCLUDE REGEX "(^|/)unix/.*")
    endif()

    if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows" AND NOT SLANG_ENABLE_DX_ON_VK)
        list(FILTER files EXCLUDE REGEX "(^|/)d3d.*/.*")
    endif()

    if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows|Linux|Darwin")
        list(FILTER files EXCLUDE REGEX "(^|/)vulkan/.*")
    endif()

    if(NOT CMAKE_SYSTEM_NAME MATCHES "Darwin")
        list(FILTER files EXCLUDE REGEX "(^|/)metal/.*")
    endif()

    if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
        list(FILTER files EXCLUDE REGEX "(^|/)open-gl/.*")
    endif()

    list(APPEND ${var} ${files})
    set(${var} ${${var}} PARENT_SCOPE)
endfunction()

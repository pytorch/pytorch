macro(caffe2_interface_library SRC DST)
  # Add an interface library definition that is dependent on the source.
  add_library(${DST} INTERFACE)
  add_dependencies(${DST} ${SRC})
  # Depending on the nature of the source library as well as the compiler,
  # determine the needed compilation flags.
  get_target_property(__tmp ${SRC} TYPE)
  # Depending on the type of the source library, we will set up the
  # link command for the specific SRC library.
  if (${__tmp} STREQUAL "STATIC_LIBRARY")
    # In the case of static library, we will need to add whole-static flags.
    if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
      target_link_libraries(
          ${DST} INTERFACE -Wl,-force_load,$<TARGET_FILE:${SRC}>)
    elseif(MSVC)
      # In MSVC, we will add whole archive in default.
      target_link_libraries(
          ${DST} INTERFACE -WHOLEARCHIVE:$<TARGET_FILE:${SRC}>)
    else()
      # Assume everything else is like gcc
      target_link_libraries(
          ${DST} INTERFACE
          -Wl,--whole-archive $<TARGET_FILE:${SRC}> -Wl,--no-whole-archive)
    endif()
  elseif(${__tmp} STREQUAL "SHARED_LIBRARY")
    if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
      target_link_libraries(
          ${DST} INTERFACE -Wl,--no-as-needed ${SRC} -Wl,--as-needed)
    else()
      target_link_libraries(${DST} INTERFACE ${SRC})
    endif()
  else()
    message(FATAL_ERROR
        "You made a CMake build file error: target " ${SRC}
        " must be of type either STATIC_LIBRARY or SHARED_LIBRARY. However, "
        "I got " ${__tmp} ".")
  endif()
  # Link all interface link libraries of the src target as well.
  target_link_libraries(${DST} INTERFACE
      $<TARGET_PROPERTY:${SRC},INTERFACE_LINK_LIBRARIES>)
  # For all other interface properties, manually inherit from the source target.
  set_target_properties(${DST} PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS
    $<TARGET_PROPERTY:${SRC},INTERFACE_COMPILE_DEFINITIONS>
    INTERFACE_COMPILE_OPTIONS
    $<TARGET_PROPERTY:${SRC},INTERFACE_COMPILE_OPTIONS>
    INTERFACE_INCLUDE_DIRECTORIES
    $<TARGET_PROPERTY:${SRC},INTERFACE_INCLUDE_DIRECTORIES>
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
    $<TARGET_PROPERTY:${SRC},INTERFACE_SYSTEM_INCLUDE_DIRECTORIES>)
endmacro()

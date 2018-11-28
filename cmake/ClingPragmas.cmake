# CMake's infuriating scoping rules: a function call creates a new
# "copy-on-write" variable scope. This means if you have a list $L in the first
# call, then recurse, $L will still be the old value it had in the first call,
# so appending to the list will create $L from before plus some new items.
# However, when you return into the previous frame, $L will be back to the value
# it had before recursing. For sanity, we just reset the list to an empty
# variable at the start of each frame, and then pass back a new list from every
# frame and concatenate it after a recursive call.
function(get_link_libraries OUTPUT_LIST TARGETS)
  set(LIBS) # Reset to a new variable
  foreach(T ${TARGETS})
    if (TARGET ${T})
      # If T is an alias to another target, resolve this alias.
      get_target_property(A ${T} ALIASED_TARGET)
      if (A)
        set(T ${A})
      endif()
      # Check if we've already been to this target.
      list(FIND VISITED_TARGETS ${T} INDEX)
      if (${INDEX} EQUAL -1)
        # Add to the visited targets, so we don't visit this target (vertex)
        # again.
        list(APPEND VISITED_TARGETS ${T})
        # If this is not an interface library, it has a physical location we
        # want to collect.
        get_target_property(TARGET_TYPE ${T} TYPE)
        if (NOT TARGET_TYPE STREQUAL "INTERFACE_LIBRARY")
          get_target_property(L ${T} LOCATION)
          list(APPEND LIBS ${L})
        endif()
        # Recurse for any link dependencies.
        get_target_property(LINK_DEPS ${T} INTERFACE_LINK_LIBRARIES)
        if (LINK_DEPS)
          get_link_libraries(D "${LINK_DEPS}")
          list(APPEND LIBS ${D})
        endif()
      endif()
    endif()
  endforeach()
  # Forward these variables to the upper scope.
  set(VISITED_TARGETS ${VISITED_TARGETS} PARENT_SCOPE)
  set(${OUTPUT_LIST} ${LIBS} PARENT_SCOPE)
endfunction()

get_link_libraries(LINK_LIB_PATHS "${TORCH_LIBRARIES}")

set(CLING_PRAGMAS_OUTPUT_FILE "${TORCH_INSTALL_PREFIX}/cling_pragmas.h")

add_custom_command(
  OUTPUT ${CLING_PRAGMAS_OUTPUT_FILE}
  COMMAND python ${TORCH_INSTALL_PREFIX}/tools/gen_cling_pragmas.py
    --include-dirs ${TORCH_INCLUDE_DIRS}
    --library-dirs ${TORCH_INSTALL_PREFIX}/lib
    --libraries ${LINK_LIB_PATHS}
    --output ${CLING_PRAGMAS_OUTPUT_FILE}
  WORKING_DIRECTORY ${TORCH_INSTALL_PREFIX})

add_custom_target(cling_pragmas DEPENDS ${CLING_PRAGMAS_OUTPUT_FILE})

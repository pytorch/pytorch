# Link CUDA libraries to the given target, i.e.: `target_link_libraries(target <args>)`
#
# Additionally makes sure CUDA stub libs don't end up being in RPath
#
# Example: link_cuda_libraries(mytarget PRIVATE ${CUDA_LIBRARIES})
function(link_cuda_libraries target)
  set(libs ${ARGN})
  set(install_rpath "$ORIGIN")
  set(filtered FALSE)
  foreach(lib IN LISTS libs)
    # CUDA stub libs are in form /prefix/lib/stubs/libcuda.so
    # So extract the name of the parent folder, to check against "stubs"
    # And the parent path which we need to add to the INSTALL_RPATH for non-stubs
    get_filename_component(parent_path "${lib}" DIRECTORY)
    get_filename_component(parent_name "${parent_path}" NAME)
    if(parent_name STREQUAL "stubs")
      message(STATUS "Filtering ${lib} from being set in ${target}'s RPATH, "
                     "because it appears to point to the CUDA stubs directory.")
      set(filtered TRUE)
    elseif(parent_path)
      list(APPEND install_rpath ${parent_path})
    endif()
  endforeach()

  # Regular link command
  target_link_libraries(${target} ${libs})
  # Manually set INSTALL_RPATH when there were any stub libs
  if(filtered)
    list(REMOVE_DUPLICATES install_rpath)
    set_target_properties(${target} PROPERTIES INSTALL_RPATH_USE_LINK_PATH FALSE)
    set_target_properties(${target} PROPERTIES INSTALL_RPATH "${install_rpath}")
  endif()
endfunction()

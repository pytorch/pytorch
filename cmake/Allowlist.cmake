if(__caffe2_allowlist_included)
  return()
endif()

set(__caffe2_allowlist_included TRUE)

set(CAFFE2_ALLOWLISTED_FILES)
if(NOT CAFFE2_ALLOWLIST)
  return()
endif()

# First read the allowlist file and break it by line.
file(READ "${CAFFE2_ALLOWLIST}" allowlist_content)
# Convert file contents into a CMake list
string(REGEX REPLACE "\n" ";" allowlist_content ${allowlist_content})

foreach(item ${allowlist_content})
  file(GLOB_RECURSE tmp ${item})
  set(CAFFE2_ALLOWLISTED_FILES ${CAFFE2_ALLOWLISTED_FILES} ${tmp})
endforeach()

macro(caffe2_do_allowlist output allowlist)
  set(_tmp)
  foreach(item ${${output}})
    list(FIND ${allowlist} ${item} _index)
    if(${_index} GREATER -1)
      set(_tmp ${_tmp} ${item})
    endif()
  endforeach()
  set(${output} ${_tmp})
endmacro()

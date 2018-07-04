
if (__caffe2_whitelist_included)
  return()
endif()

set (__caffe2_whitelist_included TRUE)

set(CAFFE2_WHITELISTED_FILES)
if (NOT CAFFE2_WHITELIST)
  return()
endif()

# First read the whitelist file and break it by line.
file(READ "${CAFFE2_WHITELIST}" whitelist_content)
# Convert file contents into a CMake list
string(REGEX REPLACE "\n" ";" whitelist_content ${whitelist_content})

foreach(item ${whitelist_content})
  file(GLOB_RECURSE tmp ${item})
  set(CAFFE2_WHITELISTED_FILES ${CAFFE2_WHITELISTED_FILES} ${tmp})
endforeach()

macro(caffe2_do_whitelist output whitelist)
  set(_tmp)
  foreach(item ${${output}})
    list(FIND ${whitelist} ${item} _index)
    if (${_index} GREATER -1)
      set(_tmp ${_tmp} ${item})
    endif()
  endforeach()
  set(${output} ${_tmp})
endmacro()

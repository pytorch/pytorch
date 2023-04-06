# CMake file to replace the string contents in Google Test and Google Mock
# Usage example:
# Patch the cmake file
#   cmake -DFILENAME=internal_utils.cmake
#         -DBACKUP=internal_utils.cmake.bak
#         -DREVERT=0
#         -P GoogleTestPatch.cmake
# Revert the changes
#   cmake -DFILENAME=internal_utils.cmake
#         -DBACKUP=internal_utils.cmake.bak
#         -DREVERT=1
#         -P GoogleTestPatch.cmake


if(REVERT)
  file(READ ${BACKUP} content)
  file(WRITE ${FILENAME} "${content}")
  file(REMOVE ${BACKUP})
else(REVERT)
  file(READ ${FILENAME} content)
  file(WRITE ${BACKUP} "${content}")
  string(REGEX REPLACE "[-/]Z[iI]" "/Z7" content "${content}")
  file(WRITE ${FILENAME} "${content}")
endif(REVERT)

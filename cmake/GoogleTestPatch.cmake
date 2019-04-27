# CMake file to replace the string contents in Google Test and Google Mock
# Usage example:
#   cmake -DFILENAME=internal_utils.cmake -P GoogleTestPatch.cmake

message("file name: ${FILENAME}")

file(READ ${FILENAME} content)

message("before change: ${content}")

string(REGEX REPLACE "[-/]Z[iI]" "/Z7" ${content} "${content}")

message("after change: ${content}")

file(WRITE ${FILENAME} "${content}")

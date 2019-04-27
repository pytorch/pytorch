# CMake file to replace the string contents in Google Test and Google Mock
# Usage example:
#   cmake -DFILENAME=internal_utils.cmake -P GoogleTestPatch.cmake

file(READ ${FILENAME} content)

string(REGEX REPLACE "[-/]Z[iI]" "/Z7" content "${content}")

file(WRITE ${FILENAME} "${content}")

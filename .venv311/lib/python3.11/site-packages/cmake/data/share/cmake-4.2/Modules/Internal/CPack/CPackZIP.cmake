# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


if(CMAKE_BINARY_DIR)
  message(FATAL_ERROR "CPackZIP.cmake may only be used by CPack internally.")
endif()

find_program(ZIP_EXECUTABLE wzzip PATHS "$ENV{ProgramFiles}/WinZip")
if(ZIP_EXECUTABLE)
  set(CPACK_ZIP_COMMAND "\"${ZIP_EXECUTABLE}\" -P \"<ARCHIVE>\" @<FILELIST>")
  set(CPACK_ZIP_NEED_QUOTES TRUE)
endif()

if(NOT ZIP_EXECUTABLE)
  find_program(ZIP_EXECUTABLE 7z PATHS "$ENV{ProgramFiles}/7-Zip")
  if(ZIP_EXECUTABLE)
    set(CPACK_ZIP_COMMAND "\"${ZIP_EXECUTABLE}\" a -tzip \"<ARCHIVE>\" @<FILELIST>")
  set(CPACK_ZIP_NEED_QUOTES TRUE)
  endif()
endif()

if(NOT ZIP_EXECUTABLE)
  find_package(Cygwin)
  find_program(ZIP_EXECUTABLE zip PATHS "${CYGWIN_INSTALL_PATH}/bin")
  if(ZIP_EXECUTABLE)
    set(CPACK_ZIP_COMMAND "\"${ZIP_EXECUTABLE}\" -r \"<ARCHIVE>\" . -i@<FILELIST>")
    set(CPACK_ZIP_NEED_QUOTES FALSE)
  endif()
endif()

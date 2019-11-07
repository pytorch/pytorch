
if(NOT "/Users/xcheng16/pytorch/android/pytorch_android/host/build-cmake/fbjni/host/googletest-download/googletest-prefix/src/googletest-stamp/googletest-gitinfo.txt" IS_NEWER_THAN "/Users/xcheng16/pytorch/android/pytorch_android/host/build-cmake/fbjni/host/googletest-download/googletest-prefix/src/googletest-stamp/googletest-gitclone-lastrun.txt")
  message(STATUS "Avoiding repeated git clone, stamp file is up to date: '/Users/xcheng16/pytorch/android/pytorch_android/host/build-cmake/fbjni/host/googletest-download/googletest-prefix/src/googletest-stamp/googletest-gitclone-lastrun.txt'")
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E remove_directory "/Users/xcheng16/pytorch/android/pytorch_android/host/build-cmake/fbjni/host/googletest-src"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/Users/xcheng16/pytorch/android/pytorch_android/host/build-cmake/fbjni/host/googletest-src'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/opt/facebook/bin/git"  clone  "https://github.com/google/googletest.git" "googletest-src"
    WORKING_DIRECTORY "/Users/xcheng16/pytorch/android/pytorch_android/host/build-cmake/fbjni/host"
    RESULT_VARIABLE error_code
    )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once:
          ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/google/googletest.git'")
endif()

execute_process(
  COMMAND "/opt/facebook/bin/git"  checkout master --
  WORKING_DIRECTORY "/Users/xcheng16/pytorch/android/pytorch_android/host/build-cmake/fbjni/host/googletest-src"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'master'")
endif()

execute_process(
  COMMAND "/opt/facebook/bin/git"  submodule update --recursive --init 
  WORKING_DIRECTORY "/Users/xcheng16/pytorch/android/pytorch_android/host/build-cmake/fbjni/host/googletest-src"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/Users/xcheng16/pytorch/android/pytorch_android/host/build-cmake/fbjni/host/googletest-src'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
    "/Users/xcheng16/pytorch/android/pytorch_android/host/build-cmake/fbjni/host/googletest-download/googletest-prefix/src/googletest-stamp/googletest-gitinfo.txt"
    "/Users/xcheng16/pytorch/android/pytorch_android/host/build-cmake/fbjni/host/googletest-download/googletest-prefix/src/googletest-stamp/googletest-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/Users/xcheng16/pytorch/android/pytorch_android/host/build-cmake/fbjni/host/googletest-download/googletest-prefix/src/googletest-stamp/googletest-gitclone-lastrun.txt'")
endif()


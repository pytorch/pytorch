# This file solely exists to let Caffe2 Android build get at the list
# of core files without having to trundle through all of ATen's CMakeLists.txt

FILE(GLOB_RECURSE ATen_CORE_HEADERS "*.h")
FILE(GLOB_RECURSE ATen_CORE_SRCS "*.cpp")
FILE(GLOB_RECURSE ATen_CORE_TEST_SRCS "*_test.cpp")
EXCLUDE(ATen_CORE_SRCS "${ATen_CORE_SRCS}" ${ATen_CORE_TEST_SRCS})

# Pass to parent
set(ATen_CORE_HEADERS ${ATen_CORE_HEADERS} PARENT_SCOPE)
set(ATen_CORE_SRCS ${ATen_CORE_SRCS} PARENT_SCOPE)
set(ATen_CORE_TEST_SRCS ${ATen_CORE_TEST_SRCS} PARENT_SCOPE)
# This is a little dodgy, because it means ALL ATen headers are made
# visible.  Fortunately, you should just get a lot of undefined symbol
# errors if you go outside core
set(ATen_CORE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../.. PARENT_SCOPE)

# This file solely exists to let Caffe2 Android build get at the list
# of core files without having to trundle through all of ATen's CMakeLists.txt

FILE(GLOB ATen_CORE_HEADERS "*.h")
FILE(GLOB ATen_CORE_SRCS "*.cpp")

# Pass to parent
set(ATen_CORE_HEADERS ${ATen_CORE_HEADERS} PARENT_SCOPE)
set(ATen_CORE_SRCS ${ATen_CORE_SRCS} PARENT_SCOPE)

# This is a platform definition file for platforms without
# an operating system using the ELF executable format.
# It is used when CMAKE_SYSTEM_NAME is set to "Generic-ELF"

include(Platform/Generic)

set(CMAKE_EXECUTABLE_SUFFIX .elf)

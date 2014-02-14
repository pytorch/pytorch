CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_MODULE_PATH}")

# Flags
# When using MSVC
IF(MSVC)
  # we want to respect the standard, and we are bored of those **** .
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE=1)
ENDIF(MSVC)

# Stay C89 strict
IF("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c89 -pedantic")
ENDIF()

# Includes
INCLUDE(TorchPaths)
INCLUDE(TorchPathsInit)
INCLUDE(TorchPackage)
INCLUDE(TorchWrap)
INCLUDE(TorchExports)

# Torch libraries
ADD_SUBDIRECTORY(lib)

INCLUDE_DIRECTORIES("${LUA_INCDIR}")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/lib/TH")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_BINARY_DIR}/lib/TH")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/lib/luaT")
LINK_DIRECTORIES("${LUA_LIBDIR}")

SET(src DiskFile.c File.c MemoryFile.c PipeFile.c Storage.c Tensor.c Timer.c utils.c init.c TensorOperator.c TensorMath.c random.c)
SET(luasrc init.lua File.lua Tensor.lua CmdLine.lua Tester.lua test/test.lua)

# Necessary do generate wrapper
ADD_TORCH_WRAP(tensormathwrap TensorMath.lua)
ADD_TORCH_WRAP(randomwrap random.lua)

ADD_TORCH_PACKAGE(torch "${src}" "${luasrc}")

TARGET_LINK_LIBRARIES(torch luaT TH)

IF(LUALIB)
  TARGET_LINK_LIBRARIES(torch ${LUALIB})
ENDIF()

INSTALL(DIRECTORY "doc" DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/torch")

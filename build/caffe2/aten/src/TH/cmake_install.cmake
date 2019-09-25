# Install script for directory: /data/users/iuriiz/pytorch2/pytorch/aten/src/TH

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/data/users/iuriiz/pytorch2/pytorch/torch")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/TH" TYPE FILE MESSAGE_NEVER FILES
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/TH.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THAllocator.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THMath.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THBlas.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THDiskFile.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THFile.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THFilePrivate.h"
    "/data/users/iuriiz/pytorch2/pytorch/build/caffe2/aten/src/TH/THGeneral.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THGenerateAllTypes.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THGenerateBFloat16Type.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THGenerateBoolType.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THGenerateDoubleType.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THGenerateFloatType.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THGenerateHalfType.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THGenerateLongType.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THGenerateIntType.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THGenerateShortType.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THGenerateCharType.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THGenerateByteType.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THGenerateFloatTypes.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THGenerateIntTypes.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THGenerateQUInt8Type.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THGenerateQInt8Type.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THGenerateQInt32Type.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THGenerateQTypes.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THLapack.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THLogAdd.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THMemoryFile.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THSize.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THStorage.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THStorageFunctions.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THTensor.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THTensorApply.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THTensorDimApply.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THVector.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THHalf.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THTensor.hpp"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THStorageFunctions.hpp"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/THGenerator.hpp"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/TH/vector" TYPE FILE MESSAGE_NEVER FILES
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/vector/AVX.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/vector/AVX2.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/../ATen/native/cpu/avx_mathfun.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/TH/generic" TYPE FILE MESSAGE_NEVER FILES
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THBlas.cpp"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THBlas.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THLapack.cpp"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THLapack.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THStorage.cpp"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THStorage.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THStorageCopy.cpp"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THStorageCopy.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THTensor.cpp"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THTensor.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THTensor.hpp"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THTensorConv.cpp"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THTensorConv.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THTensorFill.cpp"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THTensorFill.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THTensorLapack.cpp"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THTensorLapack.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THTensorMath.cpp"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THTensorMath.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THTensorRandom.cpp"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THTensorRandom.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THVectorDispatch.cpp"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THVector.h"
    "/data/users/iuriiz/pytorch2/pytorch/aten/src/TH/generic/THTensorFastGetSet.hpp"
    )
endif()


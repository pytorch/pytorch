# Install script for directory: /data/users/iuriiz/pytorch2/pytorch

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

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/Caffe2" TYPE FILE MESSAGE_NEVER FILES
    "/data/users/iuriiz/pytorch2/pytorch/build/Caffe2ConfigVersion.cmake"
    "/data/users/iuriiz/pytorch2/pytorch/build/Caffe2Config.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/Caffe2/public" TYPE FILE MESSAGE_NEVER FILES
    "/data/users/iuriiz/pytorch2/pytorch/cmake/public/cuda.cmake"
    "/data/users/iuriiz/pytorch2/pytorch/cmake/public/glog.cmake"
    "/data/users/iuriiz/pytorch2/pytorch/cmake/public/gflags.cmake"
    "/data/users/iuriiz/pytorch2/pytorch/cmake/public/mkl.cmake"
    "/data/users/iuriiz/pytorch2/pytorch/cmake/public/mkldnn.cmake"
    "/data/users/iuriiz/pytorch2/pytorch/cmake/public/protobuf.cmake"
    "/data/users/iuriiz/pytorch2/pytorch/cmake/public/threads.cmake"
    "/data/users/iuriiz/pytorch2/pytorch/cmake/public/utils.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/Caffe2/" TYPE DIRECTORY MESSAGE_NEVER FILES "/data/users/iuriiz/pytorch2/pytorch/cmake/Modules_CUDA_fix")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/Caffe2/Caffe2Targets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/Caffe2/Caffe2Targets.cmake"
         "/data/users/iuriiz/pytorch2/pytorch/build/CMakeFiles/Export/share/cmake/Caffe2/Caffe2Targets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/Caffe2/Caffe2Targets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/Caffe2/Caffe2Targets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/Caffe2" TYPE FILE MESSAGE_NEVER FILES "/data/users/iuriiz/pytorch2/pytorch/build/CMakeFiles/Export/share/cmake/Caffe2/Caffe2Targets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/Caffe2" TYPE FILE MESSAGE_NEVER FILES "/data/users/iuriiz/pytorch2/pytorch/build/CMakeFiles/Export/share/cmake/Caffe2/Caffe2Targets-release.cmake")
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/data/users/iuriiz/pytorch2/pytorch/build/third_party/protobuf/cmake/cmake_install.cmake")
  include("/data/users/iuriiz/pytorch2/pytorch/build/confu-deps/QNNPACK/cmake_install.cmake")
  include("/data/users/iuriiz/pytorch2/pytorch/build/confu-deps/pytorch_qnnpack/cmake_install.cmake")
  include("/data/users/iuriiz/pytorch2/pytorch/build/confu-deps/NNPACK/cmake_install.cmake")
  include("/data/users/iuriiz/pytorch2/pytorch/build/third_party/googletest/cmake_install.cmake")
  include("/data/users/iuriiz/pytorch2/pytorch/build/third_party/benchmark/cmake_install.cmake")
  include("/data/users/iuriiz/pytorch2/pytorch/build/third_party/gloo/cmake_install.cmake")
  include("/data/users/iuriiz/pytorch2/pytorch/build/caffe2/onnx/torch_ops/cmake_install.cmake")
  include("/data/users/iuriiz/pytorch2/pytorch/build/third_party/ideep/mkl-dnn/cmake_install.cmake")
  include("/data/users/iuriiz/pytorch2/pytorch/build/c10/cmake_install.cmake")
  include("/data/users/iuriiz/pytorch2/pytorch/build/caffe2/cmake_install.cmake")
  include("/data/users/iuriiz/pytorch2/pytorch/build/modules/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/data/users/iuriiz/pytorch2/pytorch/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")

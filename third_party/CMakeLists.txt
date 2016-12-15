# CNMEM - header + lib (CMake)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/cnmem")

# eigen - header-only, CMake
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/eigen")

# Googletest - header + lib (CMake)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/googletest")

# NCCL
if (USE_EXTERNAL_NCCL)
  externalproject_add(project_nccl
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nccl
    BUILD_COMMAND make
    CONFIGURE_COMMAND ""
    INSTALL_COMMAND ""
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/nccl
    INSTALL_DIR ${CMAKE_CURRNET_BINARY_DIR}/nccl
  )
  externalproject_get_property(project_nccl install_dir)
  add_library(nccl SHARED IMPORTED)
  set_property(TARGET nccl PROPERTY IMPORTED_LOCATION ${install_dir}/lib/libnccl.so)
  add_dependencies(nccl project_nccl)
else()
  add_library(nccl SHARED IMPORTED)
endif()

# nervanagpu
if(USE_NERVANA_GPU)
  externalproject_add(project_nervanagpu
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nervanagpu
    PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/nervanagpu
    BUILD_COMMAND make
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
  )
  externalproject_get_property(project_nervanagpu install_dir)
  add_library(nervanagpu SHARED IMPORTED)
  set_property(TARGET nervanagpu PROPERTY IMPORTED_LOCATION ${install_dir}/lib/libnervanagpu.so)
  add_dependencies(nervanagpu project_nervanagpu)
endif()

# glog


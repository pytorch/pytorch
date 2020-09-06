# ---[ RocksDB module
# In addition to being a useful module itself, RocksDB is also an exemplar
# case where show how one should built a Caffe2 module inside the Caffe2
# repository.
#
# This cmake file achieves two build modes:
# (1) If one is invoking the main Caffe2 build, we will check a USE_* option,
#     in this case USE_ROCKSDB, to test if we want to build this module.
# (2) if we are building it in a standalone way, we will find the preinstalled
#     Caffe2 library, and then build the library and install it.

# ---[ First, determine if we are building with the main repo or not.
# This is guarded by the CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO variable. It then
# routes build to two paths:
# (1) When we are building with the main repo, the caffe2_library is going to
#     be already defined, and all related paths will be defined too. So we will
#     simply test if the main repo build wants to build this module, in our
#     case by the variable "USE_ROCKSDB".
# (2) When we are not building with the main repo, we will need to do the usual
#     cmake setup: version checks, project options, find dependent packages,
#     etc.
if(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  if(NOT USE_ROCKSDB)
    return()
  endif()
else()
  cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
  project(caffe2_rocksdb CXX)
  find_package(Caffe2 REQUIRED)
  option(BUILD_SHARED_LIBS "Build shared libs." ON)
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/../../cmake/Modules)
endif()


# ---[ Second, find dependencies.
# This one should be similar to the standard dependency discovery in normal
# cmake. Note that for modules that are located in the Caffe2 repository,
# cmake related files, such as FindRocksDB in this case, should live in the
# cmake/ folder under root.
find_package(RocksDB CONFIG)
if((DEFINED RocksDB_DIR) AND RocksDB_DIR)
  list(APPEND RocksDB_LIBRARIES RocksDB::rocksdb)
else()
  message("RocksDB config not found. Fallback to legacy find.")
  find_package(RocksDB)
  if(NOT ROCKSDB_FOUND)
    message(
       FATAL_ERROR
       "RocksDB not found. If you do not need caffe2_rocksdb, set "
       "-DUSE_ROCKSDB=OFF to solve this error.")
  endif()
endif()

# ---[ Third, create the CMake target.
# The key to note is that this library will need to depend on caffe2_library,
# which is the main lib of Caffe2. If your library explicitly depends on cuda,
# then you will need to depend on the caffe2_gpu_library as well.
add_library(caffe2_rocksdb ${CMAKE_CURRENT_SOURCE_DIR}/rocksdb.cc)
target_link_libraries(caffe2_rocksdb PUBLIC torch_library)
target_link_libraries(caffe2_rocksdb PRIVATE ${RocksDB_LIBRARIES})
target_include_directories(caffe2_rocksdb PRIVATE ${RocksDB_INCLUDE_DIR})
install(TARGETS caffe2_rocksdb DESTINATION lib)

# ---[ Last, Append the library to Caffe2_MODULES, if we are building with
# the main repo.
# The purpose of this is that, for all binaries built in the Caffe2 main repo,
# they will be built with the first class modules that are built. As a result,
# these binaries will not need to explicitly load these modules before using
# them.
# Note(jiayq): this also depends on a separate cmake move to reorg test builds
# and binary builds after modules. When it is done, this note should be removed.
if(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  set(Caffe2_MODULES ${Caffe2_MODULES} caffe2_rocksdb PARENT_SCOPE)
endif()

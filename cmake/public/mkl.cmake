find_package(MKL QUIET)

if(NOT TARGET caffe2::mkl)
  add_library(caffe2::mkl INTERFACE IMPORTED)
endif()

target_include_directories(caffe2::mkl INTERFACE ${MKL_INCLUDE_DIR})
target_link_libraries(caffe2::mkl INTERFACE ${MKL_LIBRARIES})

# TODO: This is a hack, it will not pick up architecture dependent MKL libraries
# correctly; see https://github.com/pytorch/pytorch/issues/73008
foreach(MKL_LIB IN LISTS MKL_LIBRARIES)
  if(EXISTS "${MKL_LIB}")
    get_filename_component(MKL_LINK_DIR "${MKL_LIB}" DIRECTORY)
    if(IS_DIRECTORY "${MKL_LINK_DIR}")
      target_link_directories(caffe2::mkl INTERFACE "${MKL_LINK_DIR}")
    endif()
  endif()
endforeach()

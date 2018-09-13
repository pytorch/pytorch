message(STATUS "caffe2/ideep/CMakeLists.txt")
message(STATUS "caffe2/ideep/CMakeLists.txt USE_MKL is ${USE_MKL}")
if(USE_MKL AND USE_IDEEP AND CAFFE2_COMPILER_SUPPORTS_AVX2_EXTENSIONS)
  message(STATUS "Including IDEEP operators")

  # ---[ CPU files.
  file(GLOB_RECURSE avx2_srcs *.cc)
  # exclude test files and gpu files
  file(GLOB_RECURSE tmp *_test.cc)
  exclude(avx2_srcs "${avx2_srcs}" ${tmp})

  add_library(Caffe2_ideep_operators OBJECT ${avx2_srcs})
  add_dependencies(Caffe2_ideep_operators Caffe2_PROTO)
  set_target_properties(Caffe2_ideep_operators PROPERTIES COMPILE_FLAGS "-mavx2")

  # ---[ Send the lists to the parent scope.
  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS}
      $<TARGET_OBJECTS:Caffe2_ideep_operators>)
  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
elseif(USE_IDEEP AND CAFFE2_COMPILER_SUPPORTS_AVX2_EXTENSIONS)
  message(STATUS "caffe2/ideep/CMakeLists.txt would've used ideep if not for USE_MKL")
else()
  message(STATUS "Excluding ideep operators as we are not using ideep")
endif()

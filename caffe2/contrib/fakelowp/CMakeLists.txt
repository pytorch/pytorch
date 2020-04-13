if(USE_FAKELOWP)
  message(STATUS "Including FAKELOWP operators")

  # ---[ CPU files.
  file(GLOB_RECURSE tmp *.cc)
  set(FAKELOWP_CPU_SRCS ${FAKELOWP_CPU_SRCS} ${tmp})
  # exclude test files and gpu files
  file(GLOB_RECURSE tmp *_test.cc)
  exclude(FAKELOWP_CPU_SRCS "${FAKELOWP_CPU_SRCS}" ${tmp})

  # We will only build the perf kernel files if the compiler supports avx2
  # extensions.
  if(CAFFE2_COMPILER_SUPPORTS_AVX2_EXTENSIONS)
    add_library(Caffe2_fakelowp_ops STATIC ${FAKELOWP_CPU_SRCS})  
    add_dependencies(Caffe2_fakelowp_ops fbgemm Caffe2_PROTO c10)
    target_include_directories(Caffe2_fakelowp_ops BEFORE
      PRIVATE $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}/include>)
    if(MSVC AND NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      target_compile_options(Caffe2_fakelowp_ops
          PRIVATE "/arch:AVX2"
          PRIVATE "/D__FMA__"
          PRIVATE "/D__F16C__")
    else()
      target_compile_options(Caffe2_fakelowp_ops
          PRIVATE "-mavx2"
          PRIVATE "-mfma"
          PRIVATE "-mavx"
          PRIVATE "-mf16c")
    endif()
    caffe2_interface_library(
      Caffe2_fakelowp_ops Caffe2_fakelowp_ops_interface)
    list(APPEND
      Caffe2_DEPENDENCY_WHOLE_LINK_LIBS
      "Caffe2_fakelowp_ops_interface")
  endif()
else()
  message(STATUS "Excluding FakeLowP operators")
endif()

set(Caffe2_DEPENDENCY_WHOLE_LINK_LIBS
    ${Caffe2_DEPENDENCY_WHOLE_LINK_LIBS}
    PARENT_SCOPE)

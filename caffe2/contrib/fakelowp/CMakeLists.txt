if(USE_FAKELOWP)
  message(STATUS "Including FakeLowP operators")

  # ---[ CPU files.
  file(GLOB_RECURSE tmp *.cc)
  set(FAKELOWP_CPU_SRCS ${FAKELOWP_CPU_SRCS} ${tmp})
  # exclude test files and gpu files
  file(GLOB_RECURSE tmp *_test.cc)
  exclude(FAKELOWP_CPU_SRCS "${FAKELOWP_CPU_SRCS}" ${tmp})

  # We will only build the perf kernel files if the compiler supports avx2
  # extensions.
  if(CXX_AVX2_FOUND)
    add_library(caffe2_fakelowp_ops OBJECT ${FAKELOWP_CPU_SRCS})
    add_dependencies(caffe2_fakelowp_ops fbgemm cpuinfo Caffe2_PROTO c10 aten_op_header_gen)
    target_include_directories(caffe2_fakelowp_ops BEFORE
      PRIVATE $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}/include>)
    target_include_directories(caffe2_fakelowp_ops BEFORE
      PRIVATE $<BUILD_INTERFACE:${CPUINFO_SOURCE_DIR}/include>)

    if(MSVC)
      set_property(SOURCE ${FAKELOWP_CPU_SRCS}
        APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:AVX2 ")
    else()
      set_property(SOURCE ${FAKELOWP_CPU_SRCS}
        APPEND_STRING PROPERTY COMPILE_FLAGS " -mavx2 -mfma -mf16c -mxsave ")
    endif()
    set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS}
      $<TARGET_OBJECTS:caffe2_fakelowp_ops>)
  endif()
  # ---[ Send the lists to the parent scope.
  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
else()
  message(STATUS "Excluding FakeLowP operators")
endif()

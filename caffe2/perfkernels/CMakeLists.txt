if(INTERN_BUILD_MOBILE)
  list(APPEND Caffe2_CPU_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/embedding_lookup_idx.cc"
  )
  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
  return()
endif()

# ---[ CPU files.
file(GLOB common_srcs *.cc)
file(GLOB avx_srcs *_avx.cc)
file(GLOB avx2_srcs *_avx2.cc)
file(GLOB avx512_srcs *_avx512.cc)
file(GLOB sve_srcs *_sve.cc)
# exclude avx, avx2, avx512, and sve srcs from common_srcs
exclude(common_srcs "${common_srcs}" ${avx_srcs})
exclude(common_srcs "${common_srcs}" ${avx2_srcs})
exclude(common_srcs "${common_srcs}" ${avx512_srcs})
exclude(common_srcs "${common_srcs}" ${sve_srcs})

# We will always build common srcs.
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${common_srcs})

# We will only build the perf kernel files if the compiler supports avx2
# extensions.
if(CXX_AVX2_FOUND)
  add_library(Caffe2_perfkernels_avx2 STATIC ${avx2_srcs})
  target_link_libraries(Caffe2_perfkernels_avx2 PRIVATE c10)

  if(MSVC AND NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(Caffe2_perfkernels_avx2
        PRIVATE "/arch:AVX2"
        PRIVATE "/D__FMA__"
        PRIVATE "/D__F16C__")
  else()
    target_compile_options(Caffe2_perfkernels_avx2
        PRIVATE "-mavx2"
        PRIVATE "-mfma"
        PRIVATE "-mavx"
        PRIVATE "-mf16c")
  endif()
  caffe2_interface_library(
      Caffe2_perfkernels_avx2 Caffe2_perfkernels_avx2_interface)
  list(APPEND
       Caffe2_DEPENDENCY_WHOLE_LINK_LIBS
       "Caffe2_perfkernels_avx2_interface")
endif()

# We will only build the SVE perfkernel files if the compiler supports SVE
# extensions.
if(CXX_SVE_FOUND)
  add_library(Caffe2_perfkernels_sve STATIC ${sve_srcs})
  target_link_libraries(Caffe2_perfkernels_sve PRIVATE c10)
  install(TARGETS Caffe2_perfkernels_sve
      ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

  target_compile_options(Caffe2_perfkernels_sve PRIVATE "-march=armv8-a+sve")

  caffe2_interface_library(
      Caffe2_perfkernels_sve Caffe2_perfkernels_sve_interface)
  list(APPEND
       Caffe2_DEPENDENCY_WHOLE_LINK_LIBS "Caffe2_perfkernels_sve_interface")
endif()

# TODO(jiayq): currently, we only implement the very base files for the
# perfkernels. This is because to implement avx and avx2 files, we actually
# need to set up different compilation units and this is a bit more involving
# in terms of CMakefile changes. This is a stop-gap solution until we get a
# more proper implementation.

set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
set(Caffe2_DEPENDENCY_WHOLE_LINK_LIBS
    ${Caffe2_DEPENDENCY_WHOLE_LINK_LIBS}
    PARENT_SCOPE)

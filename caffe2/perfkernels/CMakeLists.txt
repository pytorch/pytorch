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
# exclude avx, avx2, and avx512 srcs from common_srcs
exclude(common_srcs "${common_srcs}" ${avx_srcs})
exclude(common_srcs "${common_srcs}" ${avx2_srcs})
exclude(common_srcs "${common_srcs}" ${avx512_srcs})

# We will always build common srcs.
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${common_srcs})

# We will only build the perf kernel files if the compiler supports avx2
# extensions.
if(CXX_AVX2_FOUND)
  add_library(Caffe2_perfkernels_avx STATIC ${avx_srcs})
  add_library(Caffe2_perfkernels_avx2 STATIC ${avx2_srcs})
  target_link_libraries(Caffe2_perfkernels_avx PRIVATE c10)
  target_link_libraries(Caffe2_perfkernels_avx2 PRIVATE c10)
  install(TARGETS Caffe2_perfkernels_avx Caffe2_perfkernels_avx2
      ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

  if(MSVC AND NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(Caffe2_perfkernels_avx
        PRIVATE "/arch:AVX"
        PRIVATE "/D__F16C__")
    target_compile_options(Caffe2_perfkernels_avx2
        PRIVATE "/arch:AVX2"
        PRIVATE "/D__FMA__"
        PRIVATE "/D__F16C__")
  else()
    target_compile_options(Caffe2_perfkernels_avx
        PRIVATE "-mavx"
        PRIVATE "-mf16c")
    target_compile_options(Caffe2_perfkernels_avx2
        PRIVATE "-mavx2"
        PRIVATE "-mfma"
        PRIVATE "-mavx"
        PRIVATE "-mf16c")
  endif()
  caffe2_interface_library(
      Caffe2_perfkernels_avx Caffe2_perfkernels_avx_interface)
  caffe2_interface_library(
      Caffe2_perfkernels_avx2 Caffe2_perfkernels_avx2_interface)
  list(APPEND
       Caffe2_DEPENDENCY_WHOLE_LINK_LIBS
       "Caffe2_perfkernels_avx_interface")
  list(APPEND
       Caffe2_DEPENDENCY_WHOLE_LINK_LIBS
       "Caffe2_perfkernels_avx2_interface")

  if(CAFFE2_COMPILER_SUPPORTS_AVX512_EXTENSIONS)
    add_library(Caffe2_perfkernels_avx512 STATIC ${avx512_srcs})
    target_link_libraries(Caffe2_perfkernels_avx512 PRIVATE c10)
    install(TARGETS Caffe2_perfkernels_avx512
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

    if(MSVC AND NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      target_compile_options(Caffe2_perfkernels_avx512
          PRIVATE "/D__AVX512F__"
          PRIVATE "/D__AVX512DQ__"
          PRIVATE "/D__AVX512VL__"
          PRIVATE "/arch:AVX2"
          PRIVATE "/D__FMA__"
          PRIVATE "/D__F16C__")
    else()
      target_compile_options(Caffe2_perfkernels_avx512
          PRIVATE "-mavx512f"
          PRIVATE "-mavx512dq"
          PRIVATE "-mavx512vl"
          PRIVATE "-mavx2"
          PRIVATE "-mfma"
          PRIVATE "-mavx"
          PRIVATE "-mf16c")
    endif()
    caffe2_interface_library(
        Caffe2_perfkernels_avx512 Caffe2_perfkernels_avx512_interface)
    list(APPEND
         Caffe2_DEPENDENCY_WHOLE_LINK_LIBS
         "Caffe2_perfkernels_avx512_interface")
  endif()
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

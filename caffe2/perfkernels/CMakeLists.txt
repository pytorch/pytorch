# ---[ CPU files.
file(GLOB common_srcs *.cc)
file(GLOB avx_srcs *_avx.cc)
file(GLOB avx2_srcs *_avx2.cc)
# exclude avx and avx2 srcs from common_srcs
exclude(common_srcs "${common_srcs}" ${avx_srcs})
exclude(common_srcs "${common_srcs}" ${avx2_srcs})

# We will always build common srcs.
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${common_srcs})

# We will only build the perf kernel files if the compiler supports avx2
# extensions.
# Currently MSVC seems to have a symbol not found error while linking (related
# to source file order?). As a result we will currently disable the perfkernel
# in msvc.
if (NOT MSVC AND CAFFE2_COMPILER_SUPPORTS_AVX2_EXTENSIONS)
  add_library(Caffe2_perfkernels_avx OBJECT ${avx_srcs})
  add_library(Caffe2_perfkernels_avx2 OBJECT ${avx2_srcs})
  add_dependencies(Caffe2_perfkernels_avx Caffe_PROTO Caffe2_PROTO)
  add_dependencies(Caffe2_perfkernels_avx2 Caffe_PROTO Caffe2_PROTO)
  if (MSVC)
    set_target_properties(
        Caffe2_perfkernels_avx PROPERTIES COMPILE_FLAGS "/arch:AVX")
    set_target_properties(
        Caffe2_perfkernels_avx2 PROPERTIES COMPILE_FLAGS "/arch:AVX2")
  else()
    set_target_properties(
        Caffe2_perfkernels_avx PROPERTIES COMPILE_FLAGS "-mavx -mf16c")
    set_target_properties(
        Caffe2_perfkernels_avx2 PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -mavx -mf16c")
  endif()
  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS}
      $<TARGET_OBJECTS:Caffe2_perfkernels_avx>
      $<TARGET_OBJECTS:Caffe2_perfkernels_avx2>)
endif()

# TODO(jiayq): currently, we only implement the very base files for the
# perfkernels. This is because to implement avx and avx2 files, we actually
# need to set up different compilation units and this is a bit more involving
# in terms of CMakefile changes. This is a stop-gap solution until we get a
# more proper implementation.

set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)

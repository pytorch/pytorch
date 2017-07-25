# ---[ CPU files.
file(GLOB common_srcs *.cc)
file(GLOB avx_srcs *_avx.cc)
file(GLOB avx2_srcs *_avx2.cc)
# exclude avx and avx2 srcs from common_srcs
exclude(common_srcs "${common_srcs}" ${avx_srcs})
exclude(common_srcs "${common_srcs}" ${avx2_srcs})

set(Caffe2_CPU_SRCS ${common_srcs} ${tmp} PARENT_SCOPE)

# TODO(jiayq): currently, we only implement the very base files for the
# perfkernels. This is because to implement avx and avx2 files, we actually
# need to set up different compilation units and this is a bit more involving
# in terms of CMakefile changes. This is a stop-gap solution until we get a
# more proper implementation.

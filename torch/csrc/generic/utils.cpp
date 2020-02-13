#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "torch/csrc/generic/utils.cpp"
#else

#if defined(TH_REAL_IS_HALF)
#define GENERATE_SPARSE 0
#else
#define GENERATE_SPARSE 1
#endif
#undef GENERATE_SPARSE

#endif

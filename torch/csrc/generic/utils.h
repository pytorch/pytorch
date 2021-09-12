#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "torch/csrc/generic/utils.h"
#else

#if defined(TH_REAL_IS_HALF)
#define GENERATE_SPARSE 0
#else
#define GENERATE_SPARSE 1
#endif

struct THPStorage;
struct THSPTensor;

typedef class THPPointer<THWStorage>      THWStoragePtr;
typedef class THPPointer<THWTensor>       THWTensorPtr;
typedef class THPPointer<THPStorage>     THPStoragePtr;

#if (!defined(THC_GENERIC_FILE)) && \
    (!defined(THQUANTIZED))
template<>
struct THPUtils_typeTraits<scalar_t> {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || \
    defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || \
    defined(THC_REAL_IS_HALF)
  static constexpr const char *python_type_str = "float";
#elif defined(TH_REAL_IS_COMPLEX) || defined(THC_REAL_IS_COMPLEX)
  static constexpr const char *python_type_str = "complex";
#else
  static constexpr const char *python_type_str = "int";
#endif
};
#endif

#undef GENERATE_SPARSE

#endif

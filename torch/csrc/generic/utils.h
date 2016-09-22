#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/utils.h"
#else

struct THPStorage;
struct THPTensor;

typedef class THPPointer<THStorage>      THStoragePtr;
typedef class THPPointer<THTensor>       THTensorPtr;
typedef class THPPointer<THPStorage>      THPStoragePtr;
typedef class THPPointer<THPTensor>       THPTensorPtr;

#if !defined(THC_GENERIC_FILE) || defined(THC_REAL_IS_HALF)
template<>
struct THPUtils_typeTraits<real> {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || \
    defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || \
    defined(THC_REAL_IS_HALF)
  static constexpr char *python_type_str = "float";
#else
  static constexpr char *python_type_str = "int";
#endif
};
#endif

#endif

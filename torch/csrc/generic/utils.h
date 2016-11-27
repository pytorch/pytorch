#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/utils.h"
#else

struct THPStorage;
struct THPTensor;
struct THSPTensor;

typedef class THPPointer<THStorage>      THStoragePtr;
typedef class THPPointer<THTensor>       THTensorPtr;
typedef class THPPointer<THPStorage>     THPStoragePtr;
typedef class THPPointer<THPTensor>      THPTensorPtr;

typedef class THPPointer<THSTensor>      THSTensorPtr;
typedef class THPPointer<THSPTensor>     THSPTensorPtr;

#if (!defined(THC_GENERIC_FILE) || defined(THC_REAL_IS_HALF)) && \
    (!defined(THD_GENERIC_FILE))
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

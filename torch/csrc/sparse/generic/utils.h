#ifndef THS_GENERIC_FILE
#define THS_GENERIC_FILE "torch/csrc/sparse/generic/utils.h"
#else

struct THSPTensor;

typedef class THPPointer<THSTensor>       THSTensorPtr;
typedef class THPPointer<THSPTensor>      THSPTensorPtr;

#endif

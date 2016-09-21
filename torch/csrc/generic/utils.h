#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/utils.h"
#else

struct THPStorage;
struct THPTensor;

typedef class THPPointer<THStorage>      THStoragePtr;
typedef class THPPointer<THTensor>       THTensorPtr;
typedef class THPPointer<THPStorage>      THPStoragePtr;
typedef class THPPointer<THPTensor>       THPTensorPtr;

#endif

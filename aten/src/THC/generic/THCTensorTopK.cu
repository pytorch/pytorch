#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorTopK.cu"
#else

#include <c10/macros/Macros.h>

void THCTensor_(topk)(THCState* state,
                      THCTensor *topK,
                      THCudaLongTensor *indices,
                      THCTensor *input_,
                      int64_t k, int dim, int dir, int sorted) {
}

#endif // THC_GENERIC_FILE

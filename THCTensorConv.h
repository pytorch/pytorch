#ifndef TH_CUDA_TENSOR_CONV_INC
#define TH_CUDA_TENSOR_CONV_INC

#include "THCTensor.h"

struct THCState;

THC_API void THCudaTensor_conv2Dmv(struct THCState *state, THCudaTensor *output,
                                   float beta, THCudaTensor *input, THCudaTensor *kernel,
                                   long srow, long scol, const char *type);
THC_API void THCudaTensor_conv2Dmm(struct THCState *state, THCudaTensor *output,
                                   float beta, THCudaTensor *input, THCudaTensor *kernel,
                                   long srow, long scol, const char *type);

THC_API void THCudaTensor_conv2DRevger(struct THCState *state, THCudaTensor *output,
                                       float beta, float alpha, THCudaTensor *input,
                                       THCudaTensor *kernel, long srow, long scol);
THC_API void THCudaTensor_conv2DRevgerm(struct THCState *state, THCudaTensor *output,
                                        float beta, float alpha, THCudaTensor *input,
                                        THCudaTensor *kernel, long srow, long scol);

THC_API void THCudaTensor_conv2Dmap(struct THCState *state, THCudaTensor *output,
                                    THCudaTensor *input, THCudaTensor *kernel,
                                    long stride_x, long stride_y, THCudaTensor *table, long fanin);

#endif

#ifndef TH_CUDA_TENSOR_CONV_INC
#define TH_CUDA_TENSOR_CONV_INC

#include "THCTensor.h"

TH_API void THCudaTensor_conv2Dmv(THCudaTensor *output, float beta, THCudaTensor *input,
                                  THCudaTensor *kernel, long srow, long scol, const char *type);
TH_API void THCudaTensor_conv2Dmm(THCudaTensor *output, float beta, THCudaTensor *input,
                                  THCudaTensor *kernel, long srow, long scol, const char *type);

TH_API void THCudaTensor_conv2DRevger(THCudaTensor *output, float beta, float alpha, 
                                      THCudaTensor *input, THCudaTensor *kernel, 
                                      long srow, long scol);
TH_API void THCudaTensor_conv2DRevgerm(THCudaTensor *output, float beta, float alpha, 
                                       THCudaTensor *input, THCudaTensor *kernel, 
                                       long srow, long scol);

TH_API void THCudaTensor_conv2Dmap(THCudaTensor *output, THCudaTensor *input,
                                  THCudaTensor *kernel, long stride_x, long stride_y
                                   , THCudaTensor *table, long fanin);

#endif

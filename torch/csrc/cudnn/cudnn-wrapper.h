#ifndef THP_CUDNN_CUDNN_WRAPPER_INC
#define THP_CUDNN_CUDNN_WRAPPER_INC

#include <cudnn.h>
#if CUDNN_MAJOR < 6
#error CuDNN v5 and lower not supported. Please update or build without it.
#endif

#endif

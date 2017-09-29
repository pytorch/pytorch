#ifndef THP_CUDNN_DESCRIPTORS_INC
#define THP_CUDNN_DESCRIPTORS_INC

#include "Exceptions.h"

#include "cudnn-wrapper.h"
#include <ATen/ATen.h>

namespace torch { namespace cudnn {

inline int dataSize(cudnnDataType_t dataType)
{
  switch (dataType) {
    case CUDNN_DATA_HALF: return 2;
    case CUDNN_DATA_FLOAT: return 4;
    default: return 8;
  }
}

inline cudnnDataType_t getDataType(const at::Tensor& t) {
  auto scalar_type = t.type().scalarType();
  if (scalar_type == at::kFloat) {
    return CUDNN_DATA_FLOAT;
  } else if (scalar_type == at::kHalf) {
    return CUDNN_DATA_HALF;
  } else if (scalar_type == at::kDouble) {
    return CUDNN_DATA_DOUBLE;
  }
  throw std::runtime_error("TensorDescriptor only supports double, float and half tensors");
}

struct TensorDescriptor
{
  cudnnTensorDescriptor_t desc;

  TensorDescriptor() : desc(NULL) {
    CHECK(cudnnCreateTensorDescriptor(&desc));
  }
  /* implicit */ TensorDescriptor(const at::Tensor& t, int pad = 0) : desc(NULL) {
    CHECK(cudnnCreateTensorDescriptor(&desc));
    set(t, pad);
  }
  TensorDescriptor(const TensorDescriptor&) = delete;
  TensorDescriptor(TensorDescriptor&& ref) { desc = ref.desc; ref.desc = NULL; }
  ~TensorDescriptor() {
    cudnnDestroyTensorDescriptor(desc);
  }
  void set(cudnnDataType_t dataType, int dim, int* size, int* stride) {
    CHECK(cudnnSetTensorNdDescriptor(desc, dataType, dim, size, stride));
  }
  void set(const at::Tensor &t, int pad = 0) {
    int dim = t.ndimension();
    if (dim > CUDNN_DIM_MAX || pad > CUDNN_DIM_MAX)
#define _STR(X) #X
#define STR(X) _STR(X)
      throw std::runtime_error("cuDNN supports only up to " STR(CUDNN_DIM_MAX) " dimensions");
#undef _STR
#undef STR
    int size[CUDNN_DIM_MAX];
    int stride[CUDNN_DIM_MAX];
    for (int i = 0; i < dim; ++i) {
      size[i] = t.size(i);
      stride[i] = t.stride(i);
    }
    for (int i = dim; i < pad; ++i) {
      size[i] = 1;
      stride[i] = 1;
    }
    dim = std::max(dim, pad);
    set(getDataType(t), dim, size, stride);
  }
};

struct FilterDescriptor
{
  cudnnFilterDescriptor_t desc;
  FilterDescriptor() : desc(NULL) {
    CHECK(cudnnCreateFilterDescriptor(&desc));
  }
  FilterDescriptor(const FilterDescriptor&) = delete;
  FilterDescriptor(FilterDescriptor&& ref)
  {
    desc = ref.desc;
    ref.desc = NULL;
  }
  ~FilterDescriptor() {
    cudnnDestroyFilterDescriptor(desc);
  }
  void set(cudnnDataType_t dataType, int dim, int* size) {
    CHECK(cudnnSetFilterNdDescriptor(desc, dataType, CUDNN_TENSOR_NCHW, dim, size));
  }
};

struct ConvolutionDescriptor
{
  cudnnConvolutionDescriptor_t desc;
  ConvolutionDescriptor() : desc(NULL) {
    CHECK(cudnnCreateConvolutionDescriptor(&desc));
  }
  ConvolutionDescriptor(const ConvolutionDescriptor&) = delete;
  ConvolutionDescriptor(ConvolutionDescriptor&& ref)
  {
    desc = ref.desc;
    ref.desc = NULL;
  }
  ~ConvolutionDescriptor() {
    cudnnDestroyConvolutionDescriptor(desc);
  }
  void set(cudnnDataType_t dataType, int dim, int* pad, int* stride, int * upscale, int groups) {
    cudnnDataType_t mathType = dataType;
    if (dataType == CUDNN_DATA_HALF) mathType = CUDNN_DATA_FLOAT;
    CHECK(cudnnSetConvolutionNdDescriptor(desc, dim, pad, stride, upscale,
                                          CUDNN_CROSS_CORRELATION, mathType));
#if CUDNN_VERSION >= 7000
    CHECK(cudnnSetConvolutionGroupCount(desc, groups));
    CHECK(cudnnSetConvolutionMathType(desc, CUDNN_DEFAULT_MATH));
    if(dataType == CUDNN_DATA_HALF)
      CHECK(cudnnSetConvolutionMathType(desc, CUDNN_TENSOR_OP_MATH));
#endif
  }
};

struct SpatialTransformerDescriptor
{
  cudnnSpatialTransformerDescriptor_t desc;
  SpatialTransformerDescriptor() : desc(NULL) {
    CHECK(cudnnCreateSpatialTransformerDescriptor(&desc));
  }
  SpatialTransformerDescriptor(const SpatialTransformerDescriptor&) = delete;
  SpatialTransformerDescriptor(SpatialTransformerDescriptor&& ref)
  {
    desc = ref.desc;
    ref.desc = NULL;
  }
  ~SpatialTransformerDescriptor() {
    cudnnDestroySpatialTransformerDescriptor(desc);
  }
  void set(cudnnDataType_t dataType, int dim, int* size) {
    CHECK(cudnnSetSpatialTransformerNdDescriptor(desc, CUDNN_SAMPLER_BILINEAR, dataType, dim, size));
  }
};

union Constant
{
  float f;
  double d;
  Constant(cudnnDataType_t dataType, double value) {
    if (dataType == CUDNN_DATA_HALF || dataType == CUDNN_DATA_FLOAT) {
      f = (float) value;
    } else {
      d = value;
    }
  }
};

}}  // namespace

#endif

#pragma once

#include "Exceptions.h"

#include "cudnn-wrapper.h"
#include <ATen/Tensor.h>
#include <ATen/Check.h>

namespace at { namespace native {

// TODO: Add constructors for all of the descriptors

inline int dataSize(cudnnDataType_t dataType)
{
  switch (dataType) {
    case CUDNN_DATA_HALF: return 2;
    case CUDNN_DATA_FLOAT: return 4;
    default: return 8;
  }
}

// The stride for a size-1 dimensions is not uniquely determined; in
// fact, it can be anything you want, because the fact that the
// tensor is size 1 at this dimension means that you will never actually
// try advancing your pointer by this stride.
//
// However, CuDNN has a much more stringent requirement on strides:
// if you are passing a contiguous input, it better be the case
// that the stride for dim i is the product of the sizes of dims
// i+1 to the end.  This stride is indeed uniquely determined.  This
// function modifies 'stride' in place so this invariant holds.
static inline void fixSizeOneDimStride(int dim, const int *size, int *stride) {
  int64_t z = 1;
  for(int d = dim-1; d >= 0; d--)
  {
    if (size[d] == 1) {
      stride[d] = z;
    } else {
      z *= size[d];
    }
  }
}

template <typename T, cudnnStatus_t (*dtor)(T*)>
struct DescriptorDeleter {
  void operator()(T* x) {
    CUDNN_CHECK(dtor(x));
  }
};

// A generic class for wrapping cuDNN descriptor types.  All you need
// is to give the underlying type the Descriptor_t points to (usually,
// if it's cudnnTensorDescriptor_t it points to cudnnTensorStruct),
// the constructor and the destructor.  Subclasses are responsible
// for forwarding constructors and defining a set() function to actually
// set the descriptor.
template <typename T, cudnnStatus_t (*ctor)(T**), cudnnStatus_t (*dtor)(T*)>
class Descriptor
{
public:
  explicit Descriptor() {
    T* raw_desc;
    CUDNN_CHECK(ctor(&raw_desc));
    desc_.reset(raw_desc);
  }

  // TODO: Figure out why const-correctness doesn't work here
  T* desc() const { return desc_.get(); }
  T* desc() { return desc_.get(); }
private:
  std::unique_ptr<T, DescriptorDeleter<T, dtor>> desc_;
};

class TensorDescriptor
  : public Descriptor<cudnnTensorStruct,
                      &cudnnCreateTensorDescriptor,
                      &cudnnDestroyTensorDescriptor>
{
public:
  explicit TensorDescriptor() : Descriptor() {};
  explicit TensorDescriptor(const at::Tensor &t, int64_t pad = 0)
    : Descriptor()
  {
    set(t, pad);
  }

  // Note [CuDNN broadcast padding]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // pad specifies the minimum dimensionality of the tensor descriptor
  // we produce (it doesn't have anything to do with, e.g., convolution
  // padding).  If 't' is lower-dimensional than 'pad', the remaining
  // dimensions (on the right) are padded with ones.  This doesn't
  // affect the underlying data layout.  This is particularly useful for
  // dealing with a pecularity of the CuDNN API, which is that broadcasting in CuDNN is
  // done in two steps: first, the client code is expected to pad out
  // (the dimensions) input tensors to be the same dimension as the
  // target broadcast, and then second, CuDNN takes of actually
  // broadcasting size 1 dimensions.
  void set(const at::Tensor &t, int64_t pad = 0);

  void print();

private:
  void set(cudnnDataType_t dataType, int dim, int* size, int* stride) {
    fixSizeOneDimStride(dim, size, stride);
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(desc(), dataType, dim, size, stride));
  }
};

std::ostream& operator<<(std::ostream & out, const TensorDescriptor& d);

class FilterDescriptor
  : public Descriptor<cudnnFilterStruct,
                      &cudnnCreateFilterDescriptor,
                      &cudnnDestroyFilterDescriptor>
{
public:
  explicit FilterDescriptor() : Descriptor() {};
  void set(const at::Tensor &t, int64_t pad = 0);

private:
  void set(cudnnDataType_t dataType, int dim, int* size) {
    CUDNN_CHECK(cudnnSetFilterNdDescriptor(desc(), dataType, CUDNN_TENSOR_NCHW, dim, size));
  }
};

struct ConvolutionDescriptor
  : public Descriptor<cudnnConvolutionStruct,
                      &cudnnCreateConvolutionDescriptor,
                      &cudnnDestroyConvolutionDescriptor>
{
  void set(cudnnDataType_t dataType, int dim, int* pad, int* stride, int * upscale /* aka dilation */, int groups) {
    cudnnDataType_t mathType = dataType;
    if (dataType == CUDNN_DATA_HALF) mathType = CUDNN_DATA_FLOAT;
    CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(desc(), dim, pad, stride, upscale,
                                          CUDNN_CROSS_CORRELATION, mathType));
#if CUDNN_VERSION >= 7000
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(desc(), groups));
    CUDNN_CHECK(cudnnSetConvolutionMathType(desc(), CUDNN_DEFAULT_MATH));
    if(dataType == CUDNN_DATA_HALF)
      CUDNN_CHECK(cudnnSetConvolutionMathType(desc(), CUDNN_TENSOR_OP_MATH));
#endif
  }
};

struct SpatialTransformerDescriptor
  : public Descriptor<cudnnSpatialTransformerStruct,
                      &cudnnCreateSpatialTransformerDescriptor,
                      &cudnnDestroySpatialTransformerDescriptor>
{
  SpatialTransformerDescriptor() : Descriptor() {
  }
  void set(cudnnDataType_t dataType, int dim, int* size) {
    CUDNN_CHECK(cudnnSetSpatialTransformerNdDescriptor(desc(), CUDNN_SAMPLER_BILINEAR, dataType, dim, size));
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

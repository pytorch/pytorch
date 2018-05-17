#pragma once

#include "Exceptions.h"

#include "miopen-wrapper.h"
#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <cuda.h>

namespace at { namespace native {

// TODO: Add constructors for all of the descriptors

inline int dataSize(miopenDataType_t dataType)
{
  switch (dataType) {
    case miopenHalf: return 2;
    case miopenFloat: return 4;
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

template <typename T, miopenStatus_t (*dtor)(T*)>
struct DescriptorDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      MIOPEN_CHECK(dtor(x));
    }
  }
};

// A generic class for wrapping cuDNN descriptor types.  All you need
// is to give the underlying type the Descriptor_t points to (usually,
// if it's miopenTensorDescriptor_t it points to miopenTensorStruct),
// the constructor and the destructor.  Subclasses are responsible
// for defining a set() function to actually set the descriptor.
//
// Descriptors default construct to a nullptr, and have a descriptor
// initialized the first time you call set() or any other initializing
// function.
template <typename T, miopenStatus_t (*ctor)(T**), miopenStatus_t (*dtor)(T*)>
class Descriptor
{
public:
  // TODO: Figure out why const-correctness doesn't work here

  // Use desc() to access the underlying descriptor pointer in
  // a read-only fashion.  Most client code should use this.
  // If the descriptor was never initialized, this will return
  // nullptr.
  T* desc() const { return desc_.get(); }
  T* desc() { return desc_.get(); }

  // Use mut_desc() to access the underlying desciptor pointer
  // if you intend to modify what it points to (e.g., using
  // miopenSetFooDescriptor).  This will ensure that the descriptor
  // is initialized.  Code in this file will use this function.
  T* mut_desc() { init(); return desc_.get(); }
protected:
  void init() {
    if (desc_ == nullptr) {
      T* raw_desc;
      MIOPEN_CHECK(ctor(&raw_desc));
      desc_.reset(raw_desc);
    }
  }
private:
  std::unique_ptr<T, DescriptorDeleter<T, dtor>> desc_;
};

class TensorDescriptor
  : public Descriptor<miopenTensorDescriptor,
                      &miopenCreateTensorDescriptor,
                      &miopenDestroyTensorDescriptor>
{
public:
  TensorDescriptor() {}
  explicit TensorDescriptor(const at::Tensor &t, size_t pad = 0) {
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

  void set(const at::Tensor &t, size_t pad = 0);
  void set(miopenDataType_t dataType, IntList sizes, IntList strides, size_t pad = 0);

  void print();

private:
  void set(miopenDataType_t dataType, int dim, int* size, int* stride) {
    fixSizeOneDimStride(dim, size, stride);
    MIOPEN_CHECK(miopenSetTensorDescriptor(mut_desc(), dataType, dim, size, stride));
    // FIXME
  }
};

std::ostream& operator<<(std::ostream & out, const TensorDescriptor& d);

class FilterDescriptor
  : public Descriptor<miopenTensorDescriptor,
                      &miopenCreateTensorDescriptor,
                      &miopenDestroyTensorDescriptor>
{
public:
  void set(const at::Tensor &t, int64_t pad = 0);

private:
  void set(miopenDataType_t dataType, int dim, int* size, int* stride) {
    MIOPEN_CHECK(miopenSetTensorDescriptor(mut_desc(), dataType, dim, size, stride));
  }
};

struct ConvolutionDescriptor
  : public Descriptor<miopenConvolutionDescriptor,
                      &miopenCreateConvolutionDescriptor,
                      &miopenDestroyConvolutionDescriptor>
{
  void set(miopenDataType_t dataType, int dim, int* pad, int* stride, int * upscale /* aka dilation */, int groups) {
    miopenDataType_t mathType = dataType;
    if (dataType == miopenHalf) mathType = miopenFloat;
    MIOPEN_CHECK(miopenInitConvolutionDescriptor(mut_desc(), miopenConvolution, *pad, *pad, *stride, *stride, 1, 1));
  }
};

#if 0
struct SpatialTransformerDescriptor
  : public Descriptor<miopenSpatialTransformerStruct,
                      &miopenCreateSpatialTransformerDescriptor,
                      &miopenDestroySpatialTransformerDescriptor>
{
  void set(miopenDataType_t dataType, int dim, int* size) {
    MIOPEN_CHECK(miopenSetSpatialTransformerNdDescriptor(mut_desc(), MIOPEN_SAMPLER_BILINEAR, dataType, dim, size));
  }
};
#endif

#if 0
struct DropoutDescriptor
  : public Descriptor<miopenDropoutStruct,
                      &miopenCreateDropoutDescriptor,
                      &miopenDestroyDropoutDescriptor>
{
  at::Tensor state;

  // Initialize a dropout descriptor's RNG state.
  // WARNING: This function is very expensive, avoid calling this function!
  // NB: it takes a Type so that we can generate a Variable if necessary
  void initialize_rng(const at::Type& ty, miopenHandle_t handle, float dropout, long long int seed) {
    AT_ASSERT(dropout > 0, "dropout must be nonzero; otherwise call set_no_dropout");
    size_t state_size;
    MIOPEN_CHECK(miopenDropoutGetStatesSize(handle, &state_size));
    AT_ASSERT(ty.is_cuda(), "dropout state type must be CUDA type");
    AT_ASSERT(ty.scalarType() == kByte, "dropout state type must be byte");
    state = ty.tensor({static_cast<int64_t>(state_size)});
    MIOPEN_CHECK(miopenSetDropoutDescriptor(mut_desc(), handle, dropout, state.data_ptr(), state_size, seed));
  }

  // Restore a dropout descriptor given a dropout probability and existing RNG state.
  // See Note [cuDNN dropout descriptor initialization]
  void set(miopenHandle_t handle, float dropout, at::Tensor state_) {
    AT_ASSERT(dropout > 0, "dropout must be nonzero; otherwise call set_no_dropout");
    state = state_;
    void *state_ptr = state.data_ptr();
    size_t state_size = state.size(0);
    // NB: The seed doesn't actually matter, so we give a dummy value
    MIOPEN_CHECK(miopenRestoreDropoutDescriptor(mut_desc(), handle, dropout, state_ptr, state_size, 0 /* seed */));
  }

  // Restore a dropout descriptor corresponding to no dropout
  // See Note [cuDNN dropout descriptor initialization]
  void set_no_dropout(miopenHandle_t handle) {
    // NB: seed doesn't matter when dropout = 0, because no random number
    // initialization actually takes place when there is no dropout.
    // NB: Empirically, miopenSetDropoutDescriptor is cheap when
    // dropoot == 0
    MIOPEN_CHECK(miopenSetDropoutDescriptor(mut_desc(), handle, 0 /* dropout */, nullptr, 0 /* state_size */, 0 /* seed */));
  }
};
#endif

//FIXME: fix dropout descriptor then enables this
#if 0
struct RNNDescriptor
  : public Descriptor<miopenRNNDescriptor,
                      &miopenCreateRNNDescriptor,
                      &miopenDestroyRNNDescriptor>
{
  DropoutDescriptor dropout_desc_;
  void set(miopenHandle_t handle, int hidden_size, int num_layers, DropoutDescriptor&& dropout_desc,
           miopenRNNInputMode_t input_mode, miopenRNNDirectionMode_t bidirectional,
           miopenRNNMode_t mode, miopenDataType_t datatype) {
    dropout_desc_ = std::move(dropout_desc);
    // FIXME
    MIOPEN_CHECK(miopenSetRNNDescriptor_v6(
          handle,
          mut_desc(),
          hidden_size,
          num_layers,
          dropout_desc_.desc(),
          input_mode,
          bidirectional,
          mode,
          MIOPEN_RNN_ALGO_STANDARD,
          datatype));
  }
};
#endif

union Constant
{
  float f;
  double d;
  Constant(miopenDataType_t dataType, double value) {
    if (dataType == miopenHalf || dataType == miopenFloat) {
      f = (float) value;
    } else {
      d = value;
    }
  }
};

}}  // namespace

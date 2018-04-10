#pragma once

#include "Exceptions.h"

#include "cudnn-wrapper.h"
#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <cuda.h>

#if CUDNN_VERSION < 7000

#include <curand_kernel.h>

/*
Note [cuDNN dropout descriptor initialization]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In most cases, setting descriptors in cuDNN is cheap (e.g.,
cudnnSetTensorNdDescriptor).  However, this is not the case for
cudnnSetDropoutDescriptor: in cuDNN 6/7 (and possibly others) it does an
expensive precomputation to initialize the random number generator states.  In
cuDNN 6, this is the ONLY official mechanism to initialize a dropout descriptor,
which means that law-abiding clients were expected to generate a dropout
descriptor once and cache it.  However, our ATen interface is (1) stateless (so
we can't cache the descriptors) and (2) does not accept arbitrary user types in
its interface (so we can't pass the descriptor in).  This puts us in a pickle.

In cuDNN 7, a new function, cudnnRestoreDropoutDescriptor was added, which
forgoes the expensive initialization process, and can initialize the
descriptor with a pre-initialized state CUDA tensor.  This is great, because
it means we can simply pass in the state tensor and then initialize the
descriptor internally.  Unfortunately, this function is not available in
cuDNN 6.

To work around this, we break the cuDNN abstraction barrier, and have
the struct layout of the underlaying dropout descriptor.  With this struct,
we can reimplement cudnnRestoreDropoutDescriptor from scratch. Great!
*/

// Reverse engineered from cuDNN 6, see Note [cuDNN dropout descriptor initialization]
struct cudnnDropoutStruct {
  float dropout;
  int nstates;
  void * states;
};

#endif

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
    if (x != nullptr) {
      CUDNN_CHECK(dtor(x));
    }
  }
};

// A generic class for wrapping cuDNN descriptor types.  All you need
// is to give the underlying type the Descriptor_t points to (usually,
// if it's cudnnTensorDescriptor_t it points to cudnnTensorStruct),
// the constructor and the destructor.  Subclasses are responsible
// for defining a set() function to actually set the descriptor.
//
// Descriptors default construct to a nullptr, and have a descriptor
// initialized the first time you call set() or any other initializing
// function.
template <typename T, cudnnStatus_t (*ctor)(T**), cudnnStatus_t (*dtor)(T*)>
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
  // cudnnSetFooDescriptor).  This will ensure that the descriptor
  // is initialized.  Code in this file will use this function.
  T* mut_desc() { init(); return desc_.get(); }
protected:
  void init() {
    if (desc_ == nullptr) {
      T* raw_desc;
      CUDNN_CHECK(ctor(&raw_desc));
      desc_.reset(raw_desc);
    }
  }
private:
  std::unique_ptr<T, DescriptorDeleter<T, dtor>> desc_;
};

class TensorDescriptor
  : public Descriptor<cudnnTensorStruct,
                      &cudnnCreateTensorDescriptor,
                      &cudnnDestroyTensorDescriptor>
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
  void set(cudnnDataType_t dataType, IntList sizes, IntList strides, size_t pad = 0);

  void print();

private:
  void set(cudnnDataType_t dataType, int dim, int* size, int* stride) {
    fixSizeOneDimStride(dim, size, stride);
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(mut_desc(), dataType, dim, size, stride));
  }
};

std::ostream& operator<<(std::ostream & out, const TensorDescriptor& d);

class FilterDescriptor
  : public Descriptor<cudnnFilterStruct,
                      &cudnnCreateFilterDescriptor,
                      &cudnnDestroyFilterDescriptor>
{
public:
  void set(const at::Tensor &t, int64_t pad = 0);

private:
  void set(cudnnDataType_t dataType, int dim, int* size) {
    CUDNN_CHECK(cudnnSetFilterNdDescriptor(mut_desc(), dataType, CUDNN_TENSOR_NCHW, dim, size));
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
    CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(mut_desc(), dim, pad, stride, upscale,
                                          CUDNN_CROSS_CORRELATION, mathType));
#if CUDNN_VERSION >= 7000
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(mut_desc(), groups));
    CUDNN_CHECK(cudnnSetConvolutionMathType(mut_desc(), CUDNN_DEFAULT_MATH));
    if(dataType == CUDNN_DATA_HALF)
      CUDNN_CHECK(cudnnSetConvolutionMathType(mut_desc(), CUDNN_TENSOR_OP_MATH));
#endif
  }
};

struct SpatialTransformerDescriptor
  : public Descriptor<cudnnSpatialTransformerStruct,
                      &cudnnCreateSpatialTransformerDescriptor,
                      &cudnnDestroySpatialTransformerDescriptor>
{
  void set(cudnnDataType_t dataType, int dim, int* size) {
    CUDNN_CHECK(cudnnSetSpatialTransformerNdDescriptor(mut_desc(), CUDNN_SAMPLER_BILINEAR, dataType, dim, size));
  }
};

#if CUDNN_VERSION < 7000

// See Note [cuDNN dropout descriptor initialization]
inline cudnnStatus_t cudnnRestoreDropoutDescriptor(
    cudnnDropoutDescriptor_t dropoutDesc,
    cudnnHandle_t handle,
    float dropout,
    void *states,
    size_t stateSizeInBytes,
    unsigned long long seed) {
  // Try to accurately simulate cuDNN's behavior, for our cuDNN 6 friends.
  // This is not entirely accurate but is good enough to catch some API
  // uses which would not be compatible in cuDNN 7.  Feel free to fix
  // this if you notice something is wrong.
  if (states == nullptr) return CUDNN_STATUS_INVALID_VALUE;
  if (stateSizeInBytes == 0) return CUDNN_STATUS_INVALID_VALUE;
  size_t expectedStateSizeInBytes;
  // State size will differ depending on size of GPU
  auto ret = cudnnDropoutGetStatesSize(handle, &expectedStateSizeInBytes);
  if (ret != CUDNN_STATUS_SUCCESS) return ret;
  if (expectedStateSizeInBytes != stateSizeInBytes) return CUDNN_STATUS_INVALID_VALUE;
  dropoutDesc->dropout = dropout;
  dropoutDesc->nstates = (int)stateSizeInBytes/sizeof(curandState_t);
  dropoutDesc->states = states;
  return CUDNN_STATUS_SUCCESS;
}

#endif // CUDNN_VERSION

struct DropoutDescriptor
  : public Descriptor<cudnnDropoutStruct,
                      &cudnnCreateDropoutDescriptor,
                      &cudnnDestroyDropoutDescriptor>
{
  at::Tensor state;

  // Initialize a dropout descriptor's RNG state.
  // WARNING: This function is very expensive, avoid calling this function!
  // NB: it takes a Type so that we can generate a Variable if necessary
  void initialize_rng(const at::Type& ty, cudnnHandle_t handle, float dropout, long long int seed) {
    AT_ASSERT(dropout > 0, "dropout must be nonzero; otherwise call set_no_dropout");
    size_t state_size;
    CUDNN_CHECK(cudnnDropoutGetStatesSize(handle, &state_size));
    AT_ASSERT(ty.is_cuda(), "dropout state type must be CUDA type");
    AT_ASSERT(ty.scalarType() == kByte, "dropout state type must be byte");
    state = ty.tensor({static_cast<int64_t>(state_size)});
    CUDNN_CHECK(cudnnSetDropoutDescriptor(mut_desc(), handle, dropout, state.data_ptr(), state_size, seed));
  }

  // Restore a dropout descriptor given a dropout probability and existing RNG state.
  // See Note [cuDNN dropout descriptor initialization]
  void set(cudnnHandle_t handle, float dropout, at::Tensor state_) {
    AT_ASSERT(dropout > 0, "dropout must be nonzero; otherwise call set_no_dropout");
    state = state_;
    void *state_ptr = state.data_ptr();
    size_t state_size = state.size(0);
    // NB: The seed doesn't actually matter, so we give a dummy value
    CUDNN_CHECK(cudnnRestoreDropoutDescriptor(mut_desc(), handle, dropout, state_ptr, state_size, 0 /* seed */));
  }

  // Restore a dropout descriptor corresponding to no dropout
  // See Note [cuDNN dropout descriptor initialization]
  void set_no_dropout(cudnnHandle_t handle) {
    // NB: seed doesn't matter when dropout = 0, because no random number
    // initialization actually takes place when there is no dropout.
    // NB: Empirically, cudnnSetDropoutDescriptor is cheap when
    // dropoot == 0
    CUDNN_CHECK(cudnnSetDropoutDescriptor(mut_desc(), handle, 0 /* dropout */, nullptr, 0 /* state_size */, 0 /* seed */));
  }
};

struct RNNDescriptor
  : public Descriptor<cudnnRNNStruct,
                      &cudnnCreateRNNDescriptor,
                      &cudnnDestroyRNNDescriptor>
{
  DropoutDescriptor dropout_desc_;
  void set(cudnnHandle_t handle, int hidden_size, int num_layers, DropoutDescriptor&& dropout_desc,
           cudnnRNNInputMode_t input_mode, cudnnDirectionMode_t bidirectional,
           cudnnRNNMode_t mode, cudnnDataType_t datatype) {
    dropout_desc_ = std::move(dropout_desc);
    CUDNN_CHECK(cudnnSetRNNDescriptor_v6(
          handle,
          mut_desc(),
          hidden_size,
          num_layers,
          dropout_desc_.desc(),
          input_mode,
          bidirectional,
          mode,
          CUDNN_RNN_ALGO_STANDARD,
          datatype));
#if CUDNN_VERSION >= 7000 && CUDA_VERSION >= 9000
    cudaDeviceProp* prop = globalContext().getCurrentDeviceProperties();
    if (prop->major >= 7) {
      if (datatype == CUDNN_DATA_HALF) {
        cudnnSetRNNMatrixMathType(mut_desc(), CUDNN_TENSOR_OP_MATH);
      } else {
        // Technically, as the default it's not necessary to explicitly
        // set this.
        cudnnSetRNNMatrixMathType(mut_desc(), CUDNN_DEFAULT_MATH);
      }
    }
#endif
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

#pragma once

#include <string>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

#include <ATen/cudnn/cudnn-wrapper.h>
#include <ATen/cudnn/Utils.h>
#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/ATenCUDAGeneral.h>
#include <cuda.h>

namespace at { namespace native {

std::string cudnnTypeToString(cudnnDataType_t dtype);

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
static inline void fixSizeOneDimStride(int dim, const int *size, int *stride, bool nhwc) {
  int64_t z = 1;
  int index = 0;
  std::vector<int> permutation(dim);

  if (nhwc) {
    permutation[index++] = 1;
  }
  for (int d = dim-1; d > 1; d--) {
    permutation[index++] = d;
  }
  if (!nhwc) {
    permutation[index++] = 1;
  }
  permutation[index++] = 0;
  for (int d : permutation) {
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
      AT_CUDNN_CHECK(dtor(x));
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
class TORCH_CUDA_CPP_API Descriptor {
 public:
  // TODO: Figure out why const-correctness doesn't work here

  // Use desc() to access the underlying descriptor pointer in
  // a read-only fashion.  Most client code should use this.
  // If the descriptor was never initialized, this will return
  // nullptr.
  T* desc() const { return desc_.get(); }
  T* desc() { return desc_.get(); }

  // Use mut_desc() to access the underlying descriptor pointer
  // if you intend to modify what it points to (e.g., using
  // cudnnSetFooDescriptor).  This will ensure that the descriptor
  // is initialized.  Code in this file will use this function.
  T* mut_desc() { init(); return desc_.get(); }
protected:
  void init() {
    if (desc_ == nullptr) {
      T* raw_desc;
      AT_CUDNN_CHECK(ctor(&raw_desc));
      desc_.reset(raw_desc);
    }
  }
private:
  std::unique_ptr<T, DescriptorDeleter<T, dtor>> desc_;
};

class TORCH_CUDA_CPP_API TensorDescriptor : public Descriptor<
                                               cudnnTensorStruct,
                                               &cudnnCreateTensorDescriptor,
                                               &cudnnDestroyTensorDescriptor> {
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
  void set(cudnnDataType_t dataType, IntArrayRef sizes, IntArrayRef strides, size_t pad = 0);

  void print();

private:
  void set(cudnnDataType_t dataType, IntArrayRef sizes, IntArrayRef strides, size_t pad, bool nhwc);

  void set(cudnnDataType_t dataType, int dim, int* size, int* stride, bool nhwc) {
    fixSizeOneDimStride(dim, size, stride, nhwc);
    AT_CUDNN_CHECK(cudnnSetTensorNdDescriptor(mut_desc(), dataType, dim, size, stride));
  }
};

std::ostream& operator<<(std::ostream & out, const TensorDescriptor& d);

class TORCH_CUDA_CPP_API FilterDescriptor : public Descriptor<
                                               cudnnFilterStruct,
                                               &cudnnCreateFilterDescriptor,
                                               &cudnnDestroyFilterDescriptor> {
 public:
  void set(const at::Tensor &t, int64_t pad = 0) {
    set(t, at::MemoryFormat::Contiguous, pad);
  }

  void set(const at::Tensor &t, const at::MemoryFormat memory_format, int64_t pad = 0);

  void print();
private:
  void set(cudnnDataType_t dataType, int dim, int* size, cudnnTensorFormat_t filter_format) {
    AT_CUDNN_CHECK(cudnnSetFilterNdDescriptor(mut_desc(), dataType, filter_format, dim, size));
  }
};

std::ostream& operator<<(std::ostream & out, const FilterDescriptor& d);

struct TORCH_CUDA_CPP_API ConvolutionDescriptor
    : public Descriptor<
          cudnnConvolutionStruct,
          &cudnnCreateConvolutionDescriptor,
          &cudnnDestroyConvolutionDescriptor> {
  void set(cudnnDataType_t dataType, int dim, int* pad, int* stride, int * upscale /* aka dilation */, int groups, bool allow_tf32) {
    cudnnDataType_t mathType = dataType;
    if (dataType == CUDNN_DATA_HALF) mathType = CUDNN_DATA_FLOAT;
    AT_CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(mut_desc(), dim, pad, stride, upscale,
                                          CUDNN_CROSS_CORRELATION, mathType));
    AT_CUDNN_CHECK(cudnnSetConvolutionGroupCount(mut_desc(), groups));
    // See Note [behavior of cudnnFind and cudnnGet]
    AT_CUDNN_CHECK(cudnnSetConvolutionMathType(mut_desc(), CUDNN_DEFAULT_MATH));
    if(dataType == CUDNN_DATA_HALF) {
      AT_CUDNN_CHECK(cudnnSetConvolutionMathType(mut_desc(), CUDNN_TENSOR_OP_MATH));
    } else if (dataType == CUDNN_DATA_FLOAT && !allow_tf32) {
#if defined(CUDNN_VERSION) && CUDNN_VERSION >= 8000
      AT_CUDNN_CHECK(cudnnSetConvolutionMathType(mut_desc(), CUDNN_FMA_MATH));
#endif
    }
  }
};

struct TORCH_CUDA_CPP_API SpatialTransformerDescriptor
    : public Descriptor<
          cudnnSpatialTransformerStruct,
          &cudnnCreateSpatialTransformerDescriptor,
          &cudnnDestroySpatialTransformerDescriptor> {
  void set(cudnnDataType_t dataType, int dim, int* size) {
    AT_CUDNN_CHECK(cudnnSetSpatialTransformerNdDescriptor(mut_desc(), CUDNN_SAMPLER_BILINEAR, dataType, dim, size));
  }
};

struct TORCH_CUDA_CPP_API DropoutDescriptor
    : public Descriptor<
          cudnnDropoutStruct,
          &cudnnCreateDropoutDescriptor,
          &cudnnDestroyDropoutDescriptor> {
  at::Tensor state;

  // Initialize a dropout descriptor's RNG state.
  // WARNING: This function is very expensive, avoid calling this function!
  void initialize_rng(cudnnHandle_t handle, float dropout, long long int seed, const TensorOptions& options) {
    TORCH_INTERNAL_ASSERT(dropout > 0, "dropout must be nonzero; otherwise call set_no_dropout");
    size_t state_size;
    AT_CUDNN_CHECK(cudnnDropoutGetStatesSize(handle, &state_size));
    AT_ASSERT(options.device().type() == kCUDA);
    AT_ASSERT(options.dtype() == kByte);
    state = at::empty({static_cast<int64_t>(state_size)}, options);
    AT_CUDNN_CHECK(cudnnSetDropoutDescriptor(mut_desc(), handle, dropout, state.data_ptr(), state_size, seed));
  }

  // Restore a dropout descriptor given a dropout probability and existing RNG state.
  void set(cudnnHandle_t handle, float dropout, at::Tensor state_) {
    TORCH_INTERNAL_ASSERT(dropout > 0, "dropout must be nonzero; otherwise call set_no_dropout");
    state = state_;
    void *state_ptr = state.data_ptr();
    size_t state_size = state.size(0);
    // NB: The seed doesn't actually matter, so we give a dummy value
    AT_CUDNN_CHECK(cudnnRestoreDropoutDescriptor(mut_desc(), handle, dropout, state_ptr, state_size, 0 /* seed */));
  }

  // Restore a dropout descriptor corresponding to no dropout
  void set_no_dropout(cudnnHandle_t handle) {
    // NB: seed doesn't matter when dropout = 0, because no random number
    // initialization actually takes place when there is no dropout.
    // NB: Empirically, cudnnSetDropoutDescriptor is cheap when
    // dropoot == 0
    AT_CUDNN_CHECK(cudnnSetDropoutDescriptor(mut_desc(), handle, 0 /* dropout */, nullptr, 0 /* state_size */, 0 /* seed */));
  }
};

struct TORCH_CUDA_CPP_API RNNDescriptor : public Descriptor<
                                             cudnnRNNStruct,
                                             &cudnnCreateRNNDescriptor,
                                             &cudnnDestroyRNNDescriptor> {
  DropoutDescriptor dropout_desc_;
  void set(cudnnHandle_t handle, int hidden_size, int proj_size, int num_layers, DropoutDescriptor&& dropout_desc,
           cudnnRNNInputMode_t input_mode, cudnnDirectionMode_t bidirectional,
           cudnnRNNMode_t mode, cudnnDataType_t datatype, cudnnDataType_t input_type, cudnnRNNAlgo_t algo, bool allow_tf32) {
    dropout_desc_ = std::move(dropout_desc);

    AT_CUDNN_CHECK(cudnnSetRNNDescriptor_v6(
          handle,
          mut_desc(),
          hidden_size,
          num_layers,
          dropout_desc_.desc(),
          input_mode,
          bidirectional,
          mode,
          algo,
          datatype));
    if (proj_size != 0) {
      AT_CUDNN_CHECK(cudnnSetRNNProjectionLayers(
            handle,
            /*rnnDesc=*/mut_desc(),
            /*recProjSize=*/proj_size,
            /*outProjSize=*/0));
    }
    cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
    if (prop->major >= 7) {
      if (input_type == CUDNN_DATA_HALF) {
        cudnnSetRNNMatrixMathType(mut_desc(), CUDNN_TENSOR_OP_MATH);
      }
#if defined(CUDNN_VERSION) && CUDNN_VERSION >= 8000
      else if (input_type == CUDNN_DATA_FLOAT && !allow_tf32) {
        cudnnSetRNNMatrixMathType(mut_desc(), CUDNN_FMA_MATH);
      }
#endif
      else {
        // Technically, as the default it's not necessary to explicitly
        // set this.
        cudnnSetRNNMatrixMathType(mut_desc(), CUDNN_DEFAULT_MATH);
      }
    }
  }
};

struct TORCH_CUDA_CPP_API CTCLossDescriptor
    : public Descriptor<
          cudnnCTCLossStruct,
          &cudnnCreateCTCLossDescriptor,
          &cudnnDestroyCTCLossDescriptor> {
  void set(cudnnDataType_t datatype) {
    AT_CUDNN_CHECK(cudnnSetCTCLossDescriptor(mut_desc(), datatype));
  }
#if CUDNN_VERSION >= 7600
  void setEx(
      cudnnDataType_t datatype,
      cudnnLossNormalizationMode_t normMode,
      cudnnNanPropagation_t gradMode) {
    AT_CUDNN_CHECK(
        cudnnSetCTCLossDescriptorEx(mut_desc(), datatype, normMode, gradMode));
  }
#endif
};

struct TORCH_CUDA_CPP_API ActivationDescriptor
    : public Descriptor<
          cudnnActivationStruct,
          &cudnnCreateActivationDescriptor,
          &cudnnDestroyActivationDescriptor> {
  void set(cudnnActivationMode_t mode) {
    AT_ASSERT(
        mode == CUDNN_ACTIVATION_RELU,
        "TODO: support more cuDNN activation modes");
    AT_CUDNN_CHECK(cudnnSetActivationDescriptor(
        mut_desc(),
        mode,
        cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
        std::numeric_limits<double>::max()));
  }
};

union Constant
{
  float f;
  double d;
  Constant(cudnnDataType_t dataType, double value) {
    if (dataType == CUDNN_DATA_HALF || dataType == CUDNN_DATA_FLOAT) {
      f = static_cast<float>(value);
    } else {
      d = value;
    }
  }
};

}}  // namespace

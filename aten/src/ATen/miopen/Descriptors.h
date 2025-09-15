#pragma once

#include <string>

#include <ATen/miopen/Exceptions.h>
#include <ATen/miopen/miopen-wrapper.h>
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <c10/macros/Export.h>

namespace at::native {

std::string miopenTypeToString(miopenDataType_t dtype);

inline int dataSize(miopenDataType_t dataType)
{
  switch (dataType) {
    case miopenHalf: return 2;
    case miopenFloat: return 4;
    case miopenBFloat16: return 2;
    default: return 8;
  }
}

// The stride for a size-1 dimensions is not uniquely determined; in
// fact, it can be anything you want, because the fact that the
// tensor is size 1 at this dimension means that you will never actually
// try advancing your pointer by this stride.
//
// We duplicate the CuDNN restriction here for MIOpen.
//
// However, CuDNN has a much more stringent requirement on strides:
// if you are passing a contiguous input, it better be the case
// that the stride for dim i is the product of the sizes of dims
// i+1 to the end.  This stride is indeed uniquely determined.  This
// function modifies 'stride' in place so this invariant holds.
template <typename T>
static inline void fixSizeOneDimStride(int dim, const T *size, T *stride, bool nhwc) {
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

template <typename T, miopenStatus_t (*dtor)(T*)>
struct DescriptorDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      MIOPEN_CHECK(dtor(x));
    }
  }
};

// A generic class for wrapping MIOpen descriptor types.  All you need
// is to give the underlying type the Descriptor_t points to (usually,
// if it's miopenTensorDescriptor_t it points to miopenTensorStruct),
// the constructor and the destructor.  Subclasses are responsible
// for defining a set() function to actually set the descriptor.
//
// Descriptors default construct to a nullptr, and have a descriptor
// initialized the first time you call set() or any other initializing
// function.
template <typename T, miopenStatus_t (*ctor)(T**), miopenStatus_t (*dtor)(T*)>
// NOLINTNEXTLINE(bugprone-exception-escape)
class TORCH_HIP_CPP_API Descriptor {
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
  // miopenSetFooDescriptor).  This will ensure that the descriptor
  // is initialized.  Code in this file will use this function.
  T* mut_desc() { init(); return desc_.get(); }
protected:
  void init() {
    if (desc_ == nullptr) {
      T* raw_desc = nullptr;
      MIOPEN_CHECK(ctor(&raw_desc));
      desc_.reset(raw_desc);
    }
  }
private:
  std::unique_ptr<T, DescriptorDeleter<T, dtor>> desc_;
};

class TORCH_HIP_CPP_API TensorDescriptor : public Descriptor<
                                               miopenTensorDescriptor,
                                               &miopenCreateTensorDescriptor,
                                               &miopenDestroyTensorDescriptor> {
 public:
  TensorDescriptor() = default;
  explicit TensorDescriptor(const at::Tensor &t, size_t pad = 0) {
    set(t, pad);
  }

  // Note [MIOpen broadcast padding]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // pad specifies the minimum dimensionality of the tensor descriptor
  // we produce (it doesn't have anything to do with, e.g., convolution
  // padding).  If 't' is lower-dimensional than 'pad', the remaining
  // dimensions (on the right) are padded with ones.  This doesn't
  // affect the underlying data layout.  This is particularly useful for
  // dealing with a peculiarity of the MIOpen API, which is that broadcasting in MIOpen is
  // done in two steps: first, the client code is expected to pad out
  // (the dimensions) input tensors to be the same dimension as the
  // target broadcast, and then second, MIOpen takes of actually
  // broadcasting size 1 dimensions.

  void set(const at::Tensor &t, size_t pad = 0);
  void set(const at::Tensor &t, at::MemoryFormat memory_format, size_t pad = 0);
  void set(miopenDataType_t dataType, IntArrayRef sizes, IntArrayRef strides, size_t pad = 0);

  void print();

private:
  void set(miopenDataType_t dataType, IntArrayRef sizes, IntArrayRef strides, size_t pad, bool nhwc);

  void set(miopenDataType_t dataType, int dim, int* size, int* stride, bool nhwc) {
    std::vector<int> strides_copy(stride, stride + dim);
    fixSizeOneDimStride<int>(dim, size, strides_copy.data(), nhwc);
    MIOPEN_CHECK(miopenSetTensorDescriptor(mut_desc(), dataType, dim, size, strides_copy.data()));
  }
};

std::ostream& operator<<(std::ostream & out, const TensorDescriptor& d);

class TORCH_HIP_CPP_API FilterDescriptor : public Descriptor<
                                               miopenTensorDescriptor,
                                               &miopenCreateTensorDescriptor,
                                               &miopenDestroyTensorDescriptor> {
 public:
  void set(const at::Tensor &t, int64_t pad = 0) {
    set(t, at::MemoryFormat::Contiguous, pad);
  }

  void set(const at::Tensor &t, const at::MemoryFormat memory_format, int64_t pad = 0);

  void print();
private:
  void set(miopenDataType_t dataType, int dim, int* size, int* stride) {
    MIOPEN_CHECK(miopenSetTensorDescriptor(mut_desc(), dataType, dim, size, stride));
  }
};

std::ostream& operator<<(std::ostream & out, const FilterDescriptor& d);

struct TORCH_HIP_CPP_API ConvolutionDescriptor
    : public Descriptor<
          miopenConvolutionDescriptor,
          &miopenCreateConvolutionDescriptor,
          &miopenDestroyConvolutionDescriptor> {
  void set(miopenDataType_t dataType,
           miopenConvolutionMode_t c_mode,
           int dim,
           int* pad,
           int* stride,
           int* upscale /* aka dilation */,
           int groups,
           bool benchmark,
           bool deterministic,
           bool allow_tf32) {
    MIOPEN_CHECK(miopenInitConvolutionNdDescriptor(mut_desc(), dim, pad, stride, upscale, c_mode));
    MIOPEN_CHECK(miopenSetConvolutionGroupCount(mut_desc(), groups));
    MIOPEN_CHECK(
        miopenSetConvolutionAttribute(mut_desc(), MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, deterministic ? 1 : 0));
    MIOPEN_CHECK(miopenSetConvolutionAttribute(
        mut_desc(), MIOPEN_CONVOLUTION_ATTRIB_MATH_TYPE, allow_tf32 ? miopenMathDefault : miopenMathPedantic));
    if (benchmark) {
      MIOPEN_CHECK(miopenSetConvolutionFindMode(mut_desc(), miopenConvolutionFindModeNormal));
    }
  }
};

// NOLINTNEXTLINE(bugprone-exception-escape)
struct TORCH_HIP_CPP_API DropoutDescriptor
    : public Descriptor<
          miopenDropoutDescriptor,
          &miopenCreateDropoutDescriptor,
          &miopenDestroyDropoutDescriptor> {
    void set(miopenHandle_t handle, float dropout, void* states, size_t stateSizeInBytes,
             unsigned long long seed, bool use_mask, bool state_evo, miopenRNGType_t rng_mode) {
      MIOPEN_CHECK(miopenSetDropoutDescriptor(mut_desc(), handle, dropout, states, stateSizeInBytes, seed, use_mask, state_evo, rng_mode));
    }

    void restore(miopenHandle_t handle, float dropout, void* states, size_t stateSizeInBytes,
      unsigned long long seed, bool use_mask, bool state_evo, miopenRNGType_t rng_mode) {
      MIOPEN_CHECK(miopenRestoreDropoutDescriptor(mut_desc(), handle, dropout, states, stateSizeInBytes, seed, use_mask, state_evo, rng_mode));
    }
};

struct TORCH_HIP_CPP_API RNNDescriptor
  : public Descriptor<miopenRNNDescriptor,
                      &miopenCreateRNNDescriptor,
                      &miopenDestroyRNNDescriptor>
{
    void set(int64_t hidden_size, int64_t num_layers, miopenRNNInputMode_t input_mode, miopenRNNDirectionMode_t direction, miopenRNNMode_t rnn_mode,
             miopenRNNBiasMode_t bias_mode, miopenRNNAlgo_t algorithm, miopenDataType_t datatype) {
      MIOPEN_CHECK(miopenSetRNNDescriptor(mut_desc(), hidden_size, num_layers, input_mode, direction, rnn_mode, bias_mode, algorithm, datatype));
    }

    void setWithDropout(DropoutDescriptor& dropout_desc, int64_t hidden_size, int64_t num_layers, miopenRNNInputMode_t input_mode, miopenRNNDirectionMode_t direction,
                        miopenRNNMode_t rnn_mode, miopenRNNBiasMode_t bias_mode, miopenRNNAlgo_t algorithm, miopenDataType_t datatype) {
      MIOPEN_CHECK(miopenSetRNNDescriptor_V2(mut_desc(), hidden_size, num_layers, dropout_desc.mut_desc(), input_mode, direction, rnn_mode, bias_mode, algorithm, datatype));
    }
};

union Constant
{
  float f;
  double d;
  Constant(miopenDataType_t dataType, double value) {
    if (dataType == miopenHalf || dataType == miopenFloat || dataType == miopenBFloat16) {
      f = static_cast<float>(value);
    } else {
      d = value;
    }
  }
};

} // namespace

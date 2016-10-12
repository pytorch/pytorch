#ifndef CAFFE2_UTILS_MKL_MKL_MEMORY_HPP_
#define CAFFE2_UTILS_MKL_MKL_MEMORY_HPP_

#include <string>
#include <vector>

#include "caffe2/utils/mkl/mkl_dnn_cppwrapper.h"

namespace caffe2 {
namespace mkl {

template <typename T>
class PrimitiveWrapper {
 public:
  PrimitiveWrapper() {}
  // Creates a primitive wrapper from an existing primitive. The wrapper
  // takes over ownership.
  explicit PrimitiveWrapper(dnnPrimitive_t primitive) : primitive_(primitive) {}
  ~PrimitiveWrapper() {
    if (primitive_) {
      MKLDNN_CHECK(dnnDelete<T>(primitive_));
    }
  }

  dnnPrimitive_t* ptr() {
    return &primitive_;
  }
  const dnnPrimitive_t& ref() const {
    return primitive_;
  }

 private:
  dnnPrimitive_t primitive_ = 0;
};

/**
 * @brief A wrapper around an opaque MKL internal resource that has certain
 * layouts and convertion primitives set up.
 */
template <typename T>
class InternalResourceWrapper {
 public:
  // Initializes an empty internal resource wrapper.
  InternalResourceWrapper() {}
  // Initialize an internal resource wrapper with the given size, strides, dnn
  // primitive and type.
  InternalResourceWrapper(
      const size_t dimension,
      const size_t size[],
      const size_t strides[],
      const dnnPrimitive_t primitive,
      const dnnResourceType_t type) {
    for (int i = 0; i < dimension; ++i) {
      total_size_ = (i == 0) ? size[i] : total_size_ * size[i];
    }
    MKLDNN_SAFE_CALL(
        dnnLayoutCreate<T>(&user_layout_, dimension, size, strides));
    MKLDNN_SAFE_CALL(
        dnnLayoutCreateFromPrimitive<T>(&layout_, primitive, type));
    MKLDNN_SAFE_CALL(
        dnnConversionCreate<T>(&convert_in_, user_layout_, layout_));
    MKLDNN_SAFE_CALL(
        dnnConversionCreate<T>(&convert_out_, layout_, user_layout_));
  }

  // Initialize an internal resource wrapper, with the size and stride
  // derived from the tensor itself.
  InternalResourceWrapper(
      const TensorCPU& tensor,
      const dnnPrimitive_t primitive,
      const dnnResourceType_t type) {
    size_t dimension = tensor.ndim();
    size_t size[dimension];
    size_t strides[dimension];
    for (int i = 0; i < dimension; ++i) {
      size[i] = tensor.dim(dimension - i - 1);
      strides[i] = (i == 0) ? 1 : strides[i - 1] * size[i - 1];
      total_size_ = (i == 0) ? size[i] : total_size_ * size[i];
    }
    MKLDNN_SAFE_CALL(
        dnnLayoutCreate<T>(&user_layout_, tensor.ndim(), size, strides));
    MKLDNN_SAFE_CALL(
        dnnLayoutCreateFromPrimitive<T>(&layout_, primitive, type));
    MKLDNN_SAFE_CALL(
        dnnConversionCreate<T>(&convert_in_, user_layout_, layout_));
    MKLDNN_SAFE_CALL(
        dnnConversionCreate<T>(&convert_out_, layout_, user_layout_));
  }

  // Destructs the internal resource wrapper.
  ~InternalResourceWrapper() {
    if (buffer_)
      MKLDNN_CHECK(dnnReleaseBuffer<T>(buffer_));
    if (layout_)
      MKLDNN_CHECK(dnnLayoutDelete<T>(layout_));
    if (convert_in_)
      MKLDNN_CHECK(dnnDelete<T>(convert_in_));
    if (convert_out_)
      MKLDNN_CHECK(dnnDelete<T>(convert_out_));
  }

  // Converts the
  void CopyIn(const void* ptr) {
    CAFFE_ENFORCE(convert_in_, "Conversion primitive not set.");
    MKLDNN_SAFE_CALL(
        dnnConversionExecute<T>(convert_in_, const_cast<void*>(ptr), buffer()));
  }

  inline void CopyIn(const TensorCPU& tensor) {
    CAFFE_ENFORCE_EQ(
        tensor.size(),
        total_size_,
        "Size does not match the expected size of the resource.");
    CopyIn(tensor.template data<T>());
  }

  void CopyOut(void* ptr) {
    CAFFE_ENFORCE(buffer_, "Canot copy out from an empty internal resource.");
    CAFFE_ENFORCE(convert_out_, "Conversion primitive not set.");
    MKLDNN_SAFE_CALL(dnnConversionExecute<T>(convert_out_, buffer_, ptr));
  }

  void CopyOut(TensorCPU* tensor) {
    CAFFE_ENFORCE_EQ(
        tensor->size(),
        total_size_,
        "CopyOut expects the output tensor size to be preset.");
    CopyOut(tensor->mutable_data<T>());
  }

  void* buffer() {
    if (!buffer_) {
      MKLDNN_SAFE_CALL(dnnAllocateBuffer<T>(&buffer_, layout_));
    }
    return buffer_;
  }

 private:
  // The internal buffer in the specific dnn layout.
  void* buffer_{nullptr};
  size_t total_size_{0};
  // The user dnn layout.
  dnnLayout_t user_layout_{0};
  // The internal dnn layout.
  dnnLayout_t layout_{0};
  // The primitive to use to convert from user layout to internal layout
  dnnPrimitive_t convert_in_{0};
  // The primitive to use to convert from internal layout to user layout
  dnnPrimitive_t convert_out_{0};
};

} // namespace mkl
} // namespace caffe2

#endif // CAFFE2_UTILS_MKL_MKL_MEMORY_HPP_

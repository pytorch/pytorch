#ifndef CAFFE2_CORE_COMMON_CUDNN_H_
#define CAFFE2_CORE_COMMON_CUDNN_H_

#include <array>
#include <mutex>

#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/types.h"

#ifndef CAFFE2_USE_CUDNN
#error("This Caffe2 install is not built with cudnn, so you should not include this file.");
#endif

#include <cudnn.h>

static_assert(
    CUDNN_VERSION >= 5000,
    "Caffe2 requires cudnn version 5.0 or above.");

#if CUDNN_VERSION < 6000
#pragma message "CUDNN version under 6.0 is supported at best effort."
#pragma message "We strongly encourage you to move to 6.0 and above."
#pragma message "This message is intended to annoy you enough to update."
#endif // CUDNN_VERSION < 6000

#define CUDNN_VERSION_MIN(major, minor, patch) \
  (CUDNN_VERSION >= ((major) * 1000 + (minor) * 100 + (patch)))

namespace caffe2 {

namespace internal {
/**
 * A helper function to obtain cudnn error strings.
 */
inline const char* cudnnGetErrorString(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "CUDNN_STATUS_LICENSE_ERROR";
    default:
      return "Unknown cudnn error number";
  }
}
} // namespace internal

// A macro that wraps around a cudnn statement so we can check if the cudnn
// execution finishes or not.
#define CUDNN_ENFORCE(condition)                          \
  do {                                                    \
    cudnnStatus_t status = condition;                     \
    CAFFE_ENFORCE_EQ(                                     \
        status,                                           \
        CUDNN_STATUS_SUCCESS,                             \
        ", Error at: ",                                   \
        __FILE__,                                         \
        ":",                                              \
        __LINE__,                                         \
        ": ",                                             \
        ::caffe2::internal::cudnnGetErrorString(status)); \
  } while (0)
#define CUDNN_CHECK(condition)                              \
  do {                                                      \
    cudnnStatus_t status = condition;                       \
    CHECK(status == CUDNN_STATUS_SUCCESS)                   \
        << ::caffe2::internal::cudnnGetErrorString(status); \
  } while (0)

// report the version of cuDNN Caffe2 was compiled with
inline size_t cudnnCompiledVersion() {
  return CUDNN_VERSION;
}
// report the runtime version of cuDNN
inline size_t cudnnRuntimeVersion() {
  return cudnnGetVersion();
}

// Check compatibility of compiled and runtime cuDNN versions
inline void CheckCuDNNVersions() {
  // Version format is major*1000 + minor*100 + patch
  // If compiled with version < 7, major, minor and patch must all match
  // If compiled with version >= 7, then either
  //    runtime_version > compiled_version
  //    major and minor match
  bool version_match = cudnnCompiledVersion() == cudnnRuntimeVersion();
  bool compiled_with_7 = cudnnCompiledVersion() >= 7000;
  bool backwards_compatible_7 = compiled_with_7 && cudnnRuntimeVersion() >= cudnnCompiledVersion();
  bool patch_compatible = compiled_with_7 && (cudnnRuntimeVersion() / 100) == (cudnnCompiledVersion() / 100);
  CAFFE_ENFORCE(version_match || backwards_compatible_7 || patch_compatible,
                "cuDNN compiled (", cudnnCompiledVersion(), ") and "
                "runtime (", cudnnRuntimeVersion(), ") versions mismatch");
}

/**
 * cudnnTypeWrapper is a wrapper class that allows us to refer to the cudnn type
 * in a template function. The class is specialized explicitly for different
 * data types below.
 */
template <typename T>
class cudnnTypeWrapper;

template <>
class cudnnTypeWrapper<float> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
  typedef const float ScalingParamType;
  typedef float BNParamType;
  static ScalingParamType* kOne() {
    static ScalingParamType v = 1.0;
    return &v;
  }
  static const ScalingParamType* kZero() {
    static ScalingParamType v = 0.0;
    return &v;
  }
};

#if CUDNN_VERSION_MIN(6, 0, 0)
template <>
class cudnnTypeWrapper<int> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_INT32;
  typedef const int ScalingParamType;
  typedef int BNParamType;
  static ScalingParamType* kOne() {
    static ScalingParamType v = 1;
    return &v;
  }
  static const ScalingParamType* kZero() {
    static ScalingParamType v = 0;
    return &v;
  }
};
#endif // CUDNN_VERSION_MIN(6, 0, 0)

template <>
class cudnnTypeWrapper<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
  typedef const double ScalingParamType;
  typedef double BNParamType;
  static ScalingParamType* kOne() {
    static ScalingParamType v = 1.0;
    return &v;
  }
  static ScalingParamType* kZero() {
    static ScalingParamType v = 0.0;
    return &v;
  }
};

template <>
class cudnnTypeWrapper<at::Half> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_HALF;
  typedef const float ScalingParamType;
  typedef float BNParamType;
  static ScalingParamType* kOne() {
    static ScalingParamType v = 1.0;
    return &v;
  }
  static ScalingParamType* kZero() {
    static ScalingParamType v = 0.0;
    return &v;
  }
};

/**
 * A wrapper function to convert the Caffe storage order to cudnn storage order
 * enum values.
 */
inline cudnnTensorFormat_t GetCudnnTensorFormat(const StorageOrder& order) {
  switch (order) {
    case StorageOrder::NHWC:
      return CUDNN_TENSOR_NHWC;
    case StorageOrder::NCHW:
      return CUDNN_TENSOR_NCHW;
    default:
      LOG(FATAL) << "Unknown cudnn equivalent for order: " << order;
  }
  // Just to suppress compiler warnings
  return CUDNN_TENSOR_NCHW;
}

/**
 * cudnnTensorDescWrapper is the placeholder that wraps around a
 * cudnnTensorDescriptor_t, allowing us to do descriptor change as-needed during
 * runtime.
 */
class cudnnTensorDescWrapper {
 public:
  cudnnTensorDescWrapper() {
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&desc_));
  }
  ~cudnnTensorDescWrapper() noexcept {
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc_));
  }

  inline cudnnTensorDescriptor_t Descriptor(
      const cudnnTensorFormat_t format,
      const cudnnDataType_t type,
      const vector<int>& dims,
      bool* changed) {
    if (type_ == type && format_ == format && dims_ == dims) {
      // if not changed, simply return the current descriptor.
      if (changed)
        *changed = false;
      return desc_;
    }
    CAFFE_ENFORCE_EQ(
        dims.size(), 4, "Currently only 4-dimensional descriptor supported.");
    format_ = format;
    type_ = type;
    dims_ = dims;
    CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
        desc_,
        format,
        type,
        dims_[0],
        (format == CUDNN_TENSOR_NCHW ? dims_[1] : dims_[3]),
        (format == CUDNN_TENSOR_NCHW ? dims_[2] : dims_[1]),
        (format == CUDNN_TENSOR_NCHW ? dims_[3] : dims_[2])));
    if (changed)
      *changed = true;
    return desc_;
  }

  template <typename T>
  inline cudnnTensorDescriptor_t Descriptor(
      const StorageOrder& order,
      const vector<int>& dims) {
    return Descriptor(
        GetCudnnTensorFormat(order), cudnnTypeWrapper<T>::type, dims, nullptr);
  }

 private:
  cudnnTensorDescriptor_t desc_;
  cudnnTensorFormat_t format_;
  cudnnDataType_t type_;
  vector<int> dims_;
  C10_DISABLE_COPY_AND_ASSIGN(cudnnTensorDescWrapper);
};

class cudnnFilterDescWrapper {
 public:
  cudnnFilterDescWrapper() {
    CUDNN_ENFORCE(cudnnCreateFilterDescriptor(&desc_));
  }
  ~cudnnFilterDescWrapper() noexcept {
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(desc_));
  }

  inline cudnnFilterDescriptor_t Descriptor(
      const StorageOrder& order,
      const cudnnDataType_t type,
      const vector<int>& dims,
      bool* changed) {
    if (type_ == type && order_ == order && dims_ == dims) {
      // if not changed, simply return the current descriptor.
      if (changed)
        *changed = false;
      return desc_;
    }
    CAFFE_ENFORCE_EQ(
        dims.size(), 4, "Currently only 4-dimensional descriptor supported.");
    order_ = order;
    type_ = type;
    dims_ = dims;
    CUDNN_ENFORCE(cudnnSetFilter4dDescriptor(
        desc_,
        type,
        GetCudnnTensorFormat(order),
        dims_[0],
        // TODO - confirm that this is correct for NHWC
        (order == StorageOrder::NCHW ? dims_[1] : dims_[3]),
        (order == StorageOrder::NCHW ? dims_[2] : dims_[1]),
        (order == StorageOrder::NCHW ? dims_[3] : dims_[2])));
    if (changed)
      *changed = true;
    return desc_;
  }

  template <typename T>
  inline cudnnFilterDescriptor_t Descriptor(
      const StorageOrder& order,
      const vector<int>& dims) {
    return Descriptor(order, cudnnTypeWrapper<T>::type, dims, nullptr);
  }

 private:
  cudnnFilterDescriptor_t desc_;
  StorageOrder order_;
  cudnnDataType_t type_;
  vector<int> dims_;
  C10_DISABLE_COPY_AND_ASSIGN(cudnnFilterDescWrapper);
};


} // namespace caffe2

#endif // CAFFE2_CORE_COMMON_CUDNN_H_

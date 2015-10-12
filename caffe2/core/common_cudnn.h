#ifndef CAFFE2_CORE_COMMON_CUDNN_H_
#define CAFFE2_CORE_COMMON_CUDNN_H_

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/types.h"
#include "caffe2/proto/caffe2.pb.h"
#include "cudnn.h"  // NOLINT
#include "caffe2/core/logging.h"

namespace caffe2 {

namespace internal {
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
  }
}
}  // namespace internal

#define CUDNN_CHECK(condition)                                                 \
  do {                                                                         \
    cudnnStatus_t status = condition;                                          \
    CAFFE_CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << " "                              \
        << "Error at: " << __FILE__ << ":" << __LINE__ << ": "                 \
        << ::caffe2::internal::cudnnGetErrorString(status);                    \
  } while (0)


template <typename T> class cudnnTypeWrapper;
template<> class cudnnTypeWrapper<float>  {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
};
template<> class cudnnTypeWrapper<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
};

inline cudnnTensorFormat_t GetCudnnTensorFormat(const StorageOrder& order) {
  switch (order) {
  case StorageOrder::NHWC:
    return CUDNN_TENSOR_NHWC;
  case StorageOrder::NCHW:
    return CUDNN_TENSOR_NCHW;
  default:
    CAFFE_LOG_FATAL << "Unknown cudnn equivalent for order: " << order;
  }
  // Just to suppress compiler warnings
  return CUDNN_TENSOR_NCHW;
}

// cudnnDescriptorMeta is the placeholder that wraps around a
// cudnnTensorDescriptor_t, allowing us to do descriptor change as-needed.
class cudnnDescriptorMeta {
 public:
  cudnnDescriptorMeta() {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc_));
  }
  cudnnDescriptorMeta(const cudnnDescriptorMeta& src) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc_));
    CAFFE_CHECK_NOTNULL(Descriptor(src.format_, src.type_, src.dims_, nullptr));
  }
  ~cudnnDescriptorMeta() {
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc_));
  }

  inline cudnnTensorDescriptor_t Descriptor(
      const cudnnTensorFormat_t format, const cudnnDataType_t type,
      const vector<int>& dims, bool* changed) {
    if (type_ == type && format_ == format && dims_ == dims) {
      // if not changed, simply return the current descriptor.
      if (changed) *changed = false;
      return desc_;
    }
    CAFFE_CHECK_EQ(dims.size(), 4)
        << "Currently only 4-dimensional descriptor supported.";
    format_ = format;
    type_ = type;
    dims_ = dims;
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        desc_, format, type, dims_[0],
        (format == CUDNN_TENSOR_NCHW? dims_[1] : dims_[3]),
        (format == CUDNN_TENSOR_NCHW? dims_[2] : dims_[1]),
        (format == CUDNN_TENSOR_NCHW? dims_[3] : dims_[2])));
    if (changed) *changed = true;
    return desc_;
  }

 private:
  cudnnTensorDescriptor_t desc_;
  cudnnTensorFormat_t format_;
  cudnnDataType_t type_;
  vector<int> dims_;
  cudnnDescriptorMeta& operator=(const cudnnDescriptorMeta&);
};

class CuDNNWrapper {
 public:
  // The default cuda context constructor.
  explicit CuDNNWrapper(CUDAContext* context)
      : cuda_context_(context), cudnn_handle_(nullptr) {}

  virtual ~CuDNNWrapper() {
    if (cudnn_handle_) {
      CUDNN_CHECK(cudnnDestroy(cudnn_handle_));
    }
  }

  cudnnHandle_t& cudnn_handle() {
    if (!cudnn_handle_) {
      CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
      CUDNN_CHECK(cudnnSetStream(
          cudnn_handle_, cuda_context_->cuda_stream()));
    }
    return cudnn_handle_;
  }

  void cudnnSetNumTensorDescriptors(int n) {
    cudnn_tensor_descriptors_.resize(n);
  }

  template <typename T>
  inline cudnnTensorDescriptor_t cudnnGetTensor4dDesc(
      const int index, const cudnnTensorFormat_t cudnn_format,
      const vector<int>& dims, bool* changed) {
    return cudnn_tensor_descriptors_.at(index).Descriptor(
        cudnn_format, cudnnTypeWrapper<T>::type, dims, changed);
  }

 protected:
  // Pointer to an external cuda context that the cudnn wrapper will use.
  CUDAContext* cuda_context_;
  cudnnHandle_t cudnn_handle_;
  std::vector<cudnnDescriptorMeta> cudnn_tensor_descriptors_;
};

}  // namespace caffe2

#endif  // CAFFE2_CORE_COMMON_CUDNN_H_

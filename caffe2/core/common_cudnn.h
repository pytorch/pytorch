#ifndef CAFFE2_CORE_COMMON_CUDNN_H_
#define CAFFE2_CORE_COMMON_CUDNN_H_

#include <mutex>

#include <cudnn.h>

#include "caffe2/core/common.h"
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/types.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/core/logging.h"

static_assert(CUDNN_VERSION >= 3000,
              "Caffe2 requires cudnn version 3.0 or above.");

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
  }
}
}  // namespace internal

// A macro that wraps around a cudnn statement so we can check if the cudnn
// execution finishes or not.
#define CUDNN_CHECK(condition)                                                 \
  do {                                                                         \
    cudnnStatus_t status = condition;                                          \
    CAFFE_CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << " "                        \
        << "Error at: " << __FILE__ << ":" << __LINE__ << ": "                 \
        << ::caffe2::internal::cudnnGetErrorString(status);                    \
  } while (0)


/**
 * cudnnTypeWrapper is a wrapper class that allows us to refer to the cudnn type
 * in a template function. The class is specialized explicitly for different
 * data types below.
 */
template <typename T> class cudnnTypeWrapper;

template<> class cudnnTypeWrapper<float> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
};

template<> class cudnnTypeWrapper<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
};

template<> class cudnnTypeWrapper<float16> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_HALF;
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
    CAFFE_LOG_FATAL << "Unknown cudnn equivalent for order: " << order;
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
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc_));
  }
  ~cudnnTensorDescWrapper() {
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
    format_ = format; type_ = type; dims_ = dims;
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        desc_, format, type, dims_[0],
        (format == CUDNN_TENSOR_NCHW? dims_[1] : dims_[3]),
        (format == CUDNN_TENSOR_NCHW? dims_[2] : dims_[1]),
        (format == CUDNN_TENSOR_NCHW? dims_[3] : dims_[2])));
    if (changed) *changed = true;
    return desc_;
  }

  template <typename T>
  inline cudnnTensorDescriptor_t Descriptor(
      const StorageOrder& order, const vector<int>& dims) {
    return Descriptor(
        GetCudnnTensorFormat(order), cudnnTypeWrapper<T>::type, dims, nullptr);
  }

 private:
  cudnnTensorDescriptor_t desc_;
  cudnnTensorFormat_t format_;
  cudnnDataType_t type_;
  vector<int> dims_;
  DISABLE_COPY_AND_ASSIGN(cudnnTensorDescWrapper);
};



class cudnnFilterDescWrapper {
 public:
  cudnnFilterDescWrapper() {
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc_));
  }
  ~cudnnFilterDescWrapper() {
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(desc_));
  }

  inline cudnnFilterDescriptor_t Descriptor(
      const StorageOrder& order, const cudnnDataType_t type,
      const vector<int>& dims, bool* changed) {
    if (type_ == type && order_ == order && dims_ == dims) {
      // if not changed, simply return the current descriptor.
      if (changed) *changed = false;
      return desc_;
    }
    CAFFE_CHECK_EQ(dims.size(), 4)
        << "Currently only 4-dimensional descriptor supported.";
    order_ = order; type_ = type; dims_ = dims;
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        desc_, type, dims_[0],
        (order == StorageOrder::NCHW? dims_[1] : dims_[3]),
        (order == StorageOrder::NCHW? dims_[2] : dims_[1]),
        (order == StorageOrder::NCHW? dims_[3] : dims_[2])));
    if (changed) *changed = true;
    return desc_;
  }

  template <typename T>
  inline cudnnFilterDescriptor_t Descriptor(
      const StorageOrder& order, const vector<int>& dims) {
    return Descriptor(order, cudnnTypeWrapper<T>::type, dims, nullptr);
  }

 private:
  cudnnFilterDescriptor_t desc_;
  StorageOrder order_;
  cudnnDataType_t type_;
  vector<int> dims_;
  DISABLE_COPY_AND_ASSIGN(cudnnFilterDescWrapper);
};


/**
 * CuDNNWrapper is a class that wraps the cudnn handles associated with a
 * specific CUDAContext.
 *
 * In caffe2, each unique CUDAContext has its own cuda stream. Since a cudnn
 * handle needs to be associated with a cuda stream, one may need to create a
 * cudnn wrapper for each CUDAContext.
 *
 * Sample usage: if you are implementing a cuda operator that uses cudnn, you
 * can have a private member like
 *     CudnnWrapper wrapper_;
 * and in your constructor, initialize it with wrapper_(device_context_).
 */
class CuDNNWrapper {
 public:
  /**
   * Creates a cudnn wrapper associated with a CUDAContext object. Note that
   * the CUDAContext object should outlive the CuDNNWrapper.
   */
  explicit CuDNNWrapper(CUDAContext* context)
      : cuda_context_(context), cudnn_handle_(nullptr) {}

  virtual ~CuDNNWrapper() {
    if (cudnn_handle_) {
      CUDNN_CHECK(cudnnDestroy(cudnn_handle_));
    }
  }

  /**
   * Returns the cudnn handle.
   */
  cudnnHandle_t& cudnn_handle() {
    if (!cudnn_handle_) {
      CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
      CUDNN_CHECK(cudnnSetStream(
          cudnn_handle_, cuda_context_->cuda_stream()));
    }
    return cudnn_handle_;
  }

 protected:
  // Pointer to an external cuda context that the cudnn wrapper will use.
  CUDAContext* cuda_context_;
  cudnnHandle_t cudnn_handle_;
};

/**
 * CuDNNWorkspaceWrapper is a wrapper class that guards a chunk of cudnn raw
 * memory used by cudnn. It provides a lock so that different potential
 * users can make sure they do not stomp on each other.
 */
class CuDNNWorkspaceWrapper {
 public:
  CuDNNWorkspaceWrapper() : data_(nullptr), nbytes_(0) {}
  ~CuDNNWorkspaceWrapper() {
    // Make sure that all usage of the workspace finishes.
    std::lock_guard<std::mutex> lock(mutex_);
    if (data_) {
      CUDAContext::Delete(data_);
    }
  }

  std::mutex& mutex() { return mutex_; }

  void* Get(const size_t nbytes) {
    if (nbytes > nbytes_) {
      if (data_) CUDAContext::Delete(data_);
      data_ = CUDAContext::New(nbytes);
      nbytes_ = nbytes;
    }
    return data_;
  }

 private:
  void* data_;
  size_t nbytes_;
  std::mutex mutex_;
};

}  // namespace caffe2

#endif  // CAFFE2_CORE_COMMON_CUDNN_H_

/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CAFFE2_CORE_COMMON_CUDNN_H_
#define CAFFE2_CORE_COMMON_CUDNN_H_

#include <array>
#include <mutex>

#include <cudnn.h>

#include "caffe2/core/common.h"
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/types.h"
#include "caffe2/proto/caffe2.pb.h"

static_assert(
    CUDNN_VERSION >= 5000,
    "Caffe2 requires cudnn version 5.0 or above.");

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
  // Major, minor and patch versions must all match
  bool version_match = cudnnCompiledVersion() == cudnnRuntimeVersion();
  CAFFE_ENFORCE(version_match,
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
class cudnnTypeWrapper<float16> {
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
  DISABLE_COPY_AND_ASSIGN(cudnnTensorDescWrapper);
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
  DISABLE_COPY_AND_ASSIGN(cudnnFilterDescWrapper);
};

class CuDNNWrapper;
/**
 * CuDNNHandles wraps around cudnnHandle_t so they can be
 * properly destructed when threads exit.
 */
class CuDNNHandles {
  friend class CuDNNWrapper;

 private:
  CuDNNHandles() {
    for (int i = 0; i < CAFFE2_COMPILE_TIME_MAX_GPUS; ++i) {
      cudnn_handle_[i] = nullptr;
    }
  }

  ~CuDNNHandles() noexcept {
    for (int i = 0; i < CAFFE2_COMPILE_TIME_MAX_GPUS; ++i) {
      if (cudnn_handle_[i]) {
        CUDNN_CHECK(cudnnDestroy(cudnn_handle_[i]));
      }
    }
  }

  cudnnHandle_t cudnn_handle_[CAFFE2_COMPILE_TIME_MAX_GPUS];
};

/**
 * CuDNNWorkspace is a wrapper around a raw cuda pointer that holds the cudnn
 * scratch space. This struct is meant to be only used in CuDNNWrapper to
 * provide a program-wide scratch space for CuDNN. The reason behind it is that
 * cudnn function calls are usually very efficient, hence one probably does not
 * want to run multiple cudnn calls at the same time. As a result, one should
 * not need more than one cudnn workspace per device.
 */
struct CuDNNWorkspace {
  ~CuDNNWorkspace() noexcept {}

  void* get(size_t nbytes) {
    if (nbytes_ < nbytes) {
      reset();
      auto data_and_deleter = CUDAContext::New(nbytes);
      data_ = {data_and_deleter.first, data_and_deleter.second};
      nbytes_ = nbytes;
    }
    CAFFE_ENFORCE_GE(nbytes_, nbytes);
    return data_.get();
  }

  void reset() {
    data_ = nullptr;
    nbytes_ = 0;
  }

 private:
  std::unique_ptr<void, MemoryDeleter> data_{nullptr, NoDelete};
  size_t nbytes_{0};
};

// CuDNNState is the owner of the CuDNNWorkspace, and serializes all
// executions of operations that use the state onto it's own stream
// (so multiple Net workers can reuse the same workspace from
// different threads and CUDA streams).
class CuDNNState {
 public:
  explicit CuDNNState(size_t gpu_id) : gpu_id_(gpu_id) {
    DeviceGuard g(gpu_id_);
    CUDNN_ENFORCE(cudnnCreate(&cudnn_handle_));
    CUDA_ENFORCE(cudaEventCreate(&before_));
    CUDA_ENFORCE(cudaEventCreate(&after_));
    CUDA_ENFORCE(cudaStreamCreate(&stream_));
    CUDNN_ENFORCE(cudnnSetStream(cudnn_handle_, stream_));
  }

  ~CuDNNState() noexcept {
    DeviceGuard g(gpu_id_);
    CUDNN_CHECK(cudnnDestroy(cudnn_handle_));
    CUDA_CHECK(cudaStreamDestroy(stream_));
    CUDA_CHECK(cudaEventDestroy(after_));
    CUDA_CHECK(cudaEventDestroy(before_));
  }

  cudnnHandle_t& cudnn_handle() {
    return cudnn_handle_;
  }

  CuDNNWorkspace& workspace() {
    return workspace_;
  }

  template <typename F>
  void execute(cudaStream_t stream, F&& f) {
    CUDA_ENFORCE(cudaEventRecord(before_, stream));
    CUDA_ENFORCE(cudaStreamWaitEvent(stream_, before_, 0));
    f(this);
    CUDA_ENFORCE(cudaEventRecord(after_, stream_));
    CUDA_ENFORCE(cudaStreamWaitEvent(stream, after_, 0));
  }

 private:
  cudnnHandle_t cudnn_handle_{nullptr};
  cudaEvent_t before_{nullptr};
  cudaEvent_t after_{nullptr};
  cudaStream_t stream_{nullptr};
  CuDNNWorkspace workspace_;
  size_t gpu_id_{0};
  DISABLE_COPY_AND_ASSIGN(CuDNNState);
};

/**
 * CuDNNWrapper is a class that wraps the cudnn handles and cudnn workspaces.
 *
 * The wrapper ensures that for each thread and each gpu, there is one
 * identical cudnn handle, which is also associated with the thread-local
 * per-device cuda stream. The wrapper also hosts the device-specific cudnn
 * workspace (scratch space for some cudnn functions).
 *
 */
class CuDNNWrapper {
 public:
  /**
   * Creates a cudnn wrapper associated with a CUDAContext object. Note that
   * the CUDAContext object should outlive the CuDNNWrapper.
   */
  explicit CuDNNWrapper(CUDAContext* context) : context_(context) {}

  /**
   * Returns the inline cudnn handle that executes on the current
   * thread's cuda_stream.
   */
  cudnnHandle_t& inline_cudnn_handle() {
    int gpu_id = context_->cuda_gpu_id();
    auto& cudnn_handle_ = tls_cudnn_handles_.cudnn_handle_[gpu_id];
    if (!cudnn_handle_) {
      context_->SwitchToDevice();
      CUDNN_ENFORCE(cudnnCreate(&cudnn_handle_));
    }
    CUDNN_ENFORCE(cudnnSetStream(cudnn_handle_, context_->cuda_stream()));
    return cudnn_handle_;
  }

  // Executes the closure F on the CuDNNState associated with state_idx
  template <typename F>
  void with_cudnn_state(size_t state_idx, F&& f) {
    CAFFE_ENFORCE(
        state_idx < CAFFE2_COMPILE_TIME_MAX_CUDNN_STATES, "Invalid state_idx");
    auto& sync_state = cudnn_states()[context_->cuda_gpu_id()][state_idx];

    DeviceGuard dg(context_->cuda_gpu_id());

    // We need to serialize execution on the CuDNNState as we can't
    // allow multiple threads to race through the cudaEventRecord
    // calls (so a worker thread might wait on another worker thread's
    // execution)
    std::lock_guard<std::mutex> g(sync_state.mutex);
    if (!sync_state.state.get()) {
      sync_state.state.reset(new CuDNNState(context_->cuda_gpu_id()));
    }
    CHECK_NOTNULL(sync_state.state.get())->execute(context_->cuda_stream(), f);
  }

 protected:
  // Pointer to an external cuda context that the cudnn wrapper will use.
  CUDAContext* context_;
  static thread_local CuDNNHandles tls_cudnn_handles_;

  static constexpr size_t CAFFE2_COMPILE_TIME_MAX_CUDNN_STATES = 4;

  struct SyncedCuDNNState {
    std::mutex mutex;
    std::unique_ptr<CuDNNState> state;
  };

  using PerGPUCuDNNStates = std::array<
      std::array<SyncedCuDNNState, CAFFE2_COMPILE_TIME_MAX_CUDNN_STATES>,
      CAFFE2_COMPILE_TIME_MAX_GPUS>;
  static PerGPUCuDNNStates& cudnn_states();

  DISABLE_COPY_AND_ASSIGN(CuDNNWrapper);
};

} // namespace caffe2

#endif // CAFFE2_CORE_COMMON_CUDNN_H_

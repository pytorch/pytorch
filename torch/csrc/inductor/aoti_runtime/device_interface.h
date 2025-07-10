#pragma once

#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef USE_XPU
#include <sycl/sycl.hpp>
#include <torch/csrc/inductor/aoti_runtime/utils_xpu.h>
#endif

#ifdef USE_MPS
#include <torch/csrc/inductor/aoti_torch/c/shim_mps.h>
#endif

#include <torch/csrc/inductor/aoti_runtime/device_utils.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

namespace torch::aot_inductor {

using RAIIDataPtr = std::unique_ptr<void, std::function<void(void*)>>;

#ifdef USE_CUDA

// NOLINTNEXTLINE(clang-diagnostic-unneeded-internal-declaration,misc-definitions-in-headers)
RAIIDataPtr RAII_cudaMalloc(size_t num_bytes) {
  void* data_ptr = nullptr;
  AOTI_RUNTIME_CUDA_CHECK(cudaMalloc((void**)&data_ptr, num_bytes));
  auto deleter = [](void* ptr) { AOTI_RUNTIME_CUDA_CHECK(cudaFree(ptr)); };
  return RAIIDataPtr(data_ptr, deleter);
}
#endif // USE_CUDA

#if defined(USE_XPU)

RAIIDataPtr RAII_xpuMalloc(size_t num_bytes) {
  sycl::queue* queue_ptr = nullptr;
  aoti_torch_get_current_sycl_queue((void**)&queue_ptr);
  void* data_ptr = sycl::malloc_device(num_bytes, *queue_ptr);
  auto deleter = [queue_ptr](void* ptr) { sycl::free(ptr, *queue_ptr); };
  return RAIIDataPtr(data_ptr, deleter);
}
#endif // USE_XPU

#if defined(USE_MPS)

RAIIDataPtr RAII_mpsMalloc(size_t num_bytes) {
  void* data_ptr = nullptr;
  aoti_torch_mps_malloc(&data_ptr, num_bytes);
  auto deleter = [](void* ptr) { aoti_torch_mps_free(ptr); };
  return RAIIDataPtr(data_ptr, deleter);
}

#endif // USE_MPS

// NOLINTNEXTLINE(misc-definitions-in-headers)
inline RAIIDataPtr RAII_cpuMalloc(size_t num_bytes) {
  void* data_ptr =
      std::malloc(num_bytes); // NOLINT(cppcoreguidelines-no-malloc)
  if (!data_ptr) {
    throw std::bad_alloc();
  }
  auto deleter = [](void* ptr) {
    std::free(ptr); // NOLINT(cppcoreguidelines-no-malloc)
  };
  return RAIIDataPtr(data_ptr, deleter);
}

// Abstract interface for device-specific operations
class AOTInductorDeviceInterface {
 public:
  virtual ~AOTInductorDeviceInterface() = default;

  // Event management
  virtual void create_event() = 0;
  virtual void destroy_event() = 0;
  virtual void record_event(void* stream) = 0;
  virtual void reset_event() = 0;

  // Memory management
  virtual RAIIDataPtr allocate_constant_blob(size_t num_bytes) = 0;
  virtual void copy_to_device(
      void* dst,
      const void* src,
      size_t size,
      size_t dst_offset = 0) = 0;

  // Copy constants and return the appropriate pointer for tensor creation
  // This handles device-specific memory layout differences
  virtual uint8_t* copy_constant_and_get_ptr(
      uint8_t* internal_ptr,
      uint8_t* constants_ptr,
      size_t constant_offset,
      const uint8_t* src_data,
      size_t bytes_read,
      size_t data_size) = 0;

  // Device management
  virtual void initialize_device(int32_t device_idx) = 0;
  virtual int32_t get_current_device() = 0;
  virtual void set_current_device(int32_t device_idx) = 0;

  // Completion status
  virtual bool is_finished() = 0;
  virtual void wait_for_completion() = 0;
};

// CUDA device implementation
#ifdef USE_CUDA
class AOTInductorCUDADevice : public AOTInductorDeviceInterface {
 private:
  std::optional<cudaEvent_t> event_;

 public:
  // Delete copy and move constructors/operators since this class manages CUDA
  // resources
  AOTInductorCUDADevice(const AOTInductorCUDADevice&) = delete;
  AOTInductorCUDADevice& operator=(const AOTInductorCUDADevice&) = delete;
  AOTInductorCUDADevice(AOTInductorCUDADevice&&) = delete;
  AOTInductorCUDADevice& operator=(AOTInductorCUDADevice&&) = delete;

  AOTInductorCUDADevice() = default;

  void destroy_event_impl() {
    if (event_) {
      auto code = cudaEventDestroy(*event_);
      if (code != cudaSuccess) {
        std::cerr << "Failed to destroy CUDA event in AOTInductor model: "
                  << cudaGetErrorString(code) << '\n';
      }
      event_.reset();
    }
  }

  ~AOTInductorCUDADevice() override {
    destroy_event_impl();
  }

  void create_event() override {
    if (!event_) {
      cudaEvent_t cuda_event = cudaEvent_t{};
      AOTI_RUNTIME_CUDA_CHECK(cudaEventCreate(&cuda_event));
      event_.emplace(cuda_event);
    }
  }

  void destroy_event() override {
    destroy_event_impl();
  }

  void record_event(void* stream) {
    if (event_) {
      auto cuda_stream = static_cast<cudaStream_t>(stream);
      AOTI_RUNTIME_CUDA_CHECK(cudaEventRecord(*event_, cuda_stream));
    }
  }

  void reset_event() override {
    // For CUDA, we don't need to reset the event, just ensure it exists
    if (!event_) {
      create_event();
    }
  }

  RAIIDataPtr allocate_constant_blob(size_t num_bytes) override {
    return RAII_cudaMalloc(num_bytes);
  }

  void copy_to_device(
      void* dst,
      const void* src,
      size_t size,
      size_t dst_offset = 0) override {
    uint8_t* dst_ptr = static_cast<uint8_t*>(dst) + dst_offset;
    AOTI_RUNTIME_CUDA_CHECK(
        cudaMemcpy(dst_ptr, src, size, cudaMemcpyHostToDevice));
  }

  void initialize_device(int32_t device_idx) override {
    if (device_idx == -1) {
      AOTI_RUNTIME_CUDA_CHECK(cudaGetDevice(&device_idx));
    } else {
      AOTI_RUNTIME_CUDA_CHECK(cudaSetDevice(device_idx));
    }
  }

  int32_t get_current_device() override {
    int32_t device_idx = 0;
    AOTI_RUNTIME_CUDA_CHECK(cudaGetDevice(&device_idx));
    return device_idx;
  }

  void set_current_device(int32_t device_idx) override {
    AOTI_RUNTIME_CUDA_CHECK(cudaSetDevice(device_idx));
  }

  uint8_t* copy_constant_and_get_ptr(
      uint8_t* internal_ptr,
      uint8_t* constants_ptr,
      size_t constant_offset,
      const uint8_t* src_data,
      size_t bytes_read,
      size_t data_size) override {
    AOTI_RUNTIME_CUDA_CHECK(cudaMemcpy(
        internal_ptr,
        src_data + bytes_read,
        data_size,
        cudaMemcpyHostToDevice));
    return internal_ptr;
  }

  bool is_finished() override {
    if (!event_) {
      throw std::runtime_error{"Model CUDA event was not initialized"};
    }
    auto event_status = cudaEventQuery(*event_);
    if (event_status == cudaSuccess) {
      return true;
    } else if (event_status == cudaErrorNotReady) {
      return false;
    }
    throw std::runtime_error(
        std::string("The model did not finish successfully. Error: ") +
        cudaGetErrorString(cudaGetLastError()));
  }

  void wait_for_completion() override {
    if (!event_) {
      throw std::runtime_error{"Model event was not initialized"};
    }
    AOTI_RUNTIME_CUDA_CHECK(cudaEventSynchronize(*event_));
  }
};
#endif // USE_CUDA

// XPU device implementation
#ifdef USE_XPU
class AOTInductorXPUDevice : public AOTInductorDeviceInterface {
 private:
  std::optional<sycl::event*> event_;

 public:
  ~AOTInductorXPUDevice() override {
    destroy_event();
  }

  void create_event() override {
    // XPU events are created when needed during record_event
  }

  void destroy_event() override {
    if (event_) {
      (*event_)->wait_and_throw();
      delete *event_;
      event_.reset();
    }
  }

  void record_event(void* stream) {
    auto* xpu_stream = static_cast<sycl::queue*>(stream);
    event_ = std::make_optional<sycl::event*>(new sycl::event(
        static_cast<sycl::queue*>(xpu_stream)->ext_oneapi_submit_barrier()));
  }

  void reset_event() override {
    if (event_) {
      (*event_)->wait_and_throw();
      delete *event_;
      event_.reset();
    }
  }

  RAIIDataPtr allocate_constant_blob(size_t num_bytes) override {
    return RAII_xpuMalloc(num_bytes);
  }

  void copy_to_device(
      void* dst,
      const void* src,
      size_t size,
      size_t dst_offset = 0) override {
    sycl::queue* queue_ptr = nullptr;
    aoti_torch_get_current_sycl_queue((void**)&queue_ptr);
    uint8_t* dst_ptr = static_cast<uint8_t*>(dst) + dst_offset;
    queue_ptr->memcpy(dst_ptr, src, size).wait();
  }

  void initialize_device(int32_t device_idx) override {
    if (device_idx == -1) {
      aoti_torch_get_current_xpu_device(&device_idx);
    } else {
      aoti_torch_set_current_xpu_device(device_idx);
    }
  }

  int32_t get_current_device() override {
    int32_t device_idx;
    aoti_torch_get_current_xpu_device(&device_idx);
    return device_idx;
  }

  void set_current_device(int32_t device_idx) override {
    aoti_torch_set_current_xpu_device(device_idx);
  }

  uint8_t* copy_constant_and_get_ptr(
      uint8_t* internal_ptr,
      uint8_t* constants_ptr,
      size_t constant_offset,
      const uint8_t* src_data,
      size_t bytes_read,
      size_t data_size) override {
    sycl::queue* queue_ptr = nullptr;
    aoti_torch_get_current_sycl_queue((void**)&queue_ptr);
    queue_ptr->memcpy(internal_ptr, src_data + bytes_read, data_size).wait();
    return internal_ptr;
  }

  bool is_finished() override {
    if (!event_) {
      throw std::runtime_error{"Model XPU event was not initialized"};
    }
    using namespace sycl::info;
    return (*event_)->get_info<event::command_execution_status>() ==
        event_command_status::complete;
  }

  void wait_for_completion() override {
    if (!event_) {
      throw std::runtime_error{"Model event was not initialized"};
    }
    (*event_)->wait_and_throw();
  }
};
#endif // USE_XPU

// MPS device implementation
#ifdef USE_MPS
class AOTInductorMPSDevice : public AOTInductorDeviceInterface {
 private:
  bool run_finished_ = false;
  int32_t device_idx_ = 0;

 public:
  void create_event() override {
    // MPS doesn't use events, just track completion with boolean
  }

  void destroy_event() override {
    // Nothing to destroy for MPS
  }

  void record_event(void* stream) {
    run_finished_ = true;
  }

  void reset_event() override {
    run_finished_ = false;
  }

  RAIIDataPtr allocate_constant_blob(size_t num_bytes) override {
    return RAII_mpsMalloc(num_bytes);
  }

  void copy_to_device(
      void* dst,
      const void* src,
      size_t size,
      size_t dst_offset = 0) override {
    // MPS uses a special memcpy function that handles the offset internally
    aoti_torch_mps_memcpy(
        static_cast<uint8_t*>(dst),
        dst_offset,
        0, // src_offset (always 0 for host source)
        size,
        static_cast<const uint8_t*>(src));
  }

  void initialize_device(int32_t device_idx) override {
    if (device_idx == -1) {
      device_idx_ = 0; //
    } else {
      device_idx_ = device_idx;
    }
  }

  int32_t get_current_device() override {
    return device_idx_;
  }

  void set_current_device(int32_t device_idx) override {
    // MPS doesn't support multiple devices
  }

  uint8_t* copy_constant_and_get_ptr(
      uint8_t* internal_ptr,
      uint8_t* constants_ptr,
      size_t constant_offset,
      const uint8_t* src_data,
      size_t bytes_read,
      size_t data_size) override {
    aoti_torch_mps_memcpy(
        constants_ptr, constant_offset, bytes_read, data_size, src_data);
    return constants_ptr;
  }

  bool is_finished() override {
    return run_finished_;
  }

  void wait_for_completion() override {
    // MPS operations are synchronous, nothing to wait for
  }
};
#endif // USE_MPS

// CPU device implementation
class AOTInductorCPUDevice : public AOTInductorDeviceInterface {
 private:
  bool run_finished_ = false;

 public:
  void create_event() override {
    // CPU doesn't use events, just track completion with boolean
  }

  void destroy_event() override {
    // Nothing to destroy for CPU
  }

  void record_event(void* stream) override {
    run_finished_ = true;
  }

  void reset_event() override {
    run_finished_ = false;
  }

  RAIIDataPtr allocate_constant_blob(size_t num_bytes) override {
    return RAII_cpuMalloc(num_bytes);
  }

  void copy_to_device(
      void* dst,
      const void* src,
      size_t size,
      size_t dst_offset = 0) override {
    uint8_t* dst_ptr = static_cast<uint8_t*>(dst) + dst_offset;
    memcpy(dst_ptr, src, size);
  }

  void initialize_device(int32_t device_idx) override {
    // CPU doesn't need device initialization
  }

  int32_t get_current_device() override {
    return 0; // CPU always uses device 0
  }

  void set_current_device(int32_t device_idx) override {
    // CPU doesn't support multiple devices
  }

  uint8_t* copy_constant_and_get_ptr(
      uint8_t* internal_ptr,
      uint8_t* constants_ptr,
      size_t constant_offset,
      const uint8_t* src_data,
      size_t bytes_read,
      size_t data_size) override {
    memcpy(internal_ptr, src_data + bytes_read, data_size);
    return internal_ptr;
  }

  bool is_finished() override {
    return run_finished_;
  }

  void wait_for_completion() override {
    // CPU operations are synchronous, nothing to wait for
  }
};

// Factory function to create appropriate device interface
inline std::unique_ptr<AOTInductorDeviceInterface> create_device_interface(
    int32_t device_type) {
#ifdef USE_CUDA
  if (device_type == aoti_torch_device_type_cuda()) {
    return std::make_unique<AOTInductorCUDADevice>();
  }
#endif
#ifdef USE_XPU
  if (device_type == aoti_torch_device_type_xpu()) {
    return std::make_unique<AOTInductorXPUDevice>();
  }
#endif
#ifdef USE_MPS
  if (device_type == aoti_torch_device_type_mps()) {
    return std::make_unique<AOTInductorMPSDevice>();
  }
#endif
  if (device_type == aoti_torch_device_type_cpu()) {
    return std::make_unique<AOTInductorCPUDevice>();
  }

  throw std::runtime_error(
      "Unsupported device type: " + std::to_string(device_type));
}

} // namespace torch::aot_inductor

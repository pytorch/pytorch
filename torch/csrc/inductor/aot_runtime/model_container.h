#pragma once

#include <future>
#include <mutex>
#include <shared_mutex>

#include <torch/csrc/inductor/aot_runtime/model.h>
#include <torch/csrc/inductor/aot_runtime/proxy_executor.h>

// At codegen time, we write out a binary file called constants.bin.
// We then turn the raw binary to an object file that exposes this
// symbol and link it into the final .so.
// For information on the binary format, see `man objcopy`, under
// the "binary-architecture" flag:
// https://man7.org/linux/man-pages/man1/objcopy.1.html
// todo: use #embed in C++ 23 once available
extern const uint8_t _binary_constants_bin_start[];
extern const uint8_t _binary_constants_bin_end[];

#define AOT_CONST_GPU_ALIGNMENT 64

namespace {

using CUDAPtr = std::unique_ptr<void, std::function<void(void*)>>;

CUDAPtr RAII_cudaMalloc(size_t num_bytes) {
  void* data_ptr;
  C10_CUDA_CHECK(cudaMalloc((void**)&data_ptr, num_bytes));
  auto deleter = [](void* ptr) { C10_CUDA_CHECK(cudaFree(ptr)); };
  return CUDAPtr(data_ptr, deleter);
}
} // anonymous namespace

namespace torch {
namespace aot_inductor {

class AOTInductorModelContainer {
 public:
  AOTInductorModelContainer(size_t num_models, bool is_cpu = false) {
    LOG(INFO) << "Constructing an AOTInductorModelContainer with " << num_models
              << " model instances";
    TORCH_CHECK(num_models > 0, "expected num_models to be larger than 0");

    constants_ = std::make_shared<ConstantMap>();
    models_.reserve(num_models);
    available_models_.reserve(num_models);
    for (size_t i = 0; i < num_models; ++i) {
      models_.push_back(AOTInductorModel::Create(constants_));
      available_models_.push_back(models_.back().get());
    }

    // Note that the all following fields (input_names_, output_names
    // and max_output_shapes_) can be filled in by the AOT
    // codegen. However, we choose to query such information from
    // the owned AOTInductorModel for a couple of reasons:
    //   * simplify the codegen templates
    //   * reduce information fragmentation and duplication
    //   * the initialization process below is done only once when the container
    //     is constructed, so it would have little performance impact
    auto* model = available_models_[0];
    size_t num_inputs = model->num_inputs();
    input_names_.reserve(num_inputs);
    input_dtypes_.reserve(num_inputs);
    max_input_shapes_.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; i++) {
      input_names_.push_back(model->input_name(i));
      input_dtypes_.push_back(model->get_input_dtype(i));
      max_input_shapes_.push_back(model->max_input_shape(i));
    }

    size_t num_outputs = model->num_outputs();
    output_names_.reserve(num_outputs);
    output_dtypes_.reserve(num_outputs);
    max_output_shapes_.reserve(num_outputs);
    output_shapes_.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; i++) {
      output_names_.push_back(model->output_name(i));
      output_dtypes_.push_back(model->get_output_dtype(i));
      max_output_shapes_.push_back(model->max_output_shape(i));
      output_shapes_.emplace_back(max_output_shapes_.back().size(), -1);
    }

    size_t num_constants = model->num_constants();
    constants_->reserve(num_constants);

    std::vector<size_t> constants_internal_offset(num_constants);
    if (!is_cpu) {
      // Compute required blob size with 64-alignment if on GPU.
      size_t max_blob = 0;
      for (size_t i = 0; i < num_constants; i++) {
        size_t data_size = model->constant_data_size(i);
        if (data_size % AOT_CONST_GPU_ALIGNMENT) {
          data_size = AOT_CONST_GPU_ALIGNMENT +
              (data_size / AOT_CONST_GPU_ALIGNMENT) * AOT_CONST_GPU_ALIGNMENT;
        }
        constants_internal_offset[i] = max_blob;
        max_blob += data_size;
      }
      constant_blob_ = RAII_cudaMalloc(max_blob);
    }

    size_t bytes_read = 0;
    for (size_t i = 0; i < num_constants; i++) {
      std::string name = model->constant_name(i);
      size_t data_size = model->constant_data_size(i);
      uint8_t* internal_ptr;
      if (is_cpu) {
        // get pointer to constant which is packed in model during compile time.
        internal_ptr =
            const_cast<uint8_t*>(_binary_constants_bin_start) + bytes_read;
      } else {
        auto* constants_ptr = static_cast<uint8_t*>(constant_blob_.get());
        internal_ptr = constants_ptr + constants_internal_offset[i];
        // Copy data to GPU memory
        // TODO: Handle shared storage case.
        C10_CUDA_CHECK(cudaMemcpy(
            internal_ptr,
            _binary_constants_bin_start + bytes_read,
            data_size,
            cudaMemcpyHostToDevice));
      }
      bytes_read += data_size;

      // Create at::Tensor from copied memory.
      auto dtype = model->constant_type(i);
      auto size = model->constant_shape(i);
      auto stride = model->constant_stride(i);
      auto offset = model->constant_offset(i);

      auto device_type = at::kCUDA;
      if (is_cpu) {
        device_type = at::kCPU;
      }
      auto tensor = at::for_blob(internal_ptr, size)
                        .strides(stride)
                        .storage_offset(offset)
                        .options(at::device(device_type).dtype(dtype))
                        .make_tensor();
      constants_->emplace(std::move(name), std::move(tensor));
    }
  }

  void run(
      const std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<int64_t>>** output_shapes,
      cudaStream_t stream,
      ProxyExecutor* proxy_executor) {
    auto* model = get_available_model();
    try {
      AOT_VECTOR_SIZE_CHECK(inputs, num_inputs());
      AOT_VECTOR_SIZE_CHECK(outputs, num_outputs());
      model->run(inputs, outputs, stream, proxy_executor);
    } catch (...) {
      std::lock_guard lk(models_mutex_);
      available_models_.push_back(model);
      throw;
    }

    for (size_t i = 0; i < num_outputs(); i++) {
      output_shapes_[i] = model->output_shape(i);
    }
    *output_shapes = &output_shapes_;

    {
      std::lock_guard lk(models_mutex_);
      pending_models_.push_back(model);
    }
    pending_models_available_.notify_one();
  }

  size_t num_inputs() const {
    return input_names_.size();
  }

  size_t num_outputs() const {
    return output_names_.size();
  }

  const char* input_name(size_t idx) const {
    return input_names_.at(idx).c_str();
  }

  const char* output_name(size_t idx) const {
    return output_names_.at(idx).c_str();
  }

  const char* get_input_dtype(size_t idx) const {
    return input_dtypes_.at(idx).c_str();
  }

  const char* get_output_dtype(size_t idx) const {
    return output_dtypes_.at(idx).c_str();
  }

  size_t num_models() const {
    return models_.size();
  }

  const std::vector<int64_t>& max_input_shape(size_t idx) const {
    return max_input_shapes_[idx];
  }

  const std::vector<int64_t>& max_output_shape(size_t idx) const {
    return max_output_shapes_[idx];
  }

 private:
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<std::string> input_dtypes_;
  std::vector<std::string> output_dtypes_;
  // Holds the upper-bound value for each dimension of any input shape.
  std::vector<std::vector<int64_t>> max_input_shapes_;

  // Holds the upper-bound value for each dimension of any output shape.
  std::vector<std::vector<int64_t>> max_output_shapes_;

  // Holds the blob storage for constants' at::Tensor.
  CUDAPtr constant_blob_;

  // Holds the mapping of constants to at::Tensor.
  // The underlying data of at::Tensor is in constant_blob_.
  std::shared_ptr<ConstantMap> constants_;

  // Holds the current value for each dimension of any output shape.
  std::vector<std::vector<int64_t>> output_shapes_;

  // Holds all the AOTInductorModel instances owned by this container.
  std::vector<std::unique_ptr<AOTInductorModel>> models_;

  // Holds the AOTInductorModel instances available for inference.
  std::vector<AOTInductorModel*> available_models_;

  // Holds the AOTInductorModel instances that have started running
  // inference and can be placed onto available_models_ upon their
  // completion.
  std::deque<AOTInductorModel*> pending_models_;

  // Protects available_models_ and pending_models_.
  std::mutex models_mutex_;

  // Notified whenever a model is placed onto pending_models_.
  std::condition_variable pending_models_available_;

  AOTInductorModel* get_available_model() {
    std::unique_lock lk(models_mutex_);
    if (available_models_.empty()) {
      reclaim_finished_models(lk);
    }
    auto* result = available_models_.back();
    available_models_.pop_back();
    return result;
  }

  void reclaim_finished_models(std::unique_lock<std::mutex>& lk) {
    // push finished model instances to the end of pending_models_
    auto it = std::stable_partition(
        pending_models_.begin(),
        pending_models_.end(),
        [](AOTInductorModel* m) { return !m->is_finished(); });

    if (it != pending_models_.end()) {
      // We have finished model instances that can be pushed into
      // available_models_ so that we don't have to be blocked on waiting
      // the pending_models_available_ condition.
      available_models_.insert(
          available_models_.end(), it, pending_models_.end());
      pending_models_.erase(it, pending_models_.end());
      return;
    }

    pending_models_available_.wait(
        lk, [this]() { return !pending_models_.empty(); });
    // Let's make the schedule simple first. We always wait on the first
    // pending_models_ to be complete.
    auto* model = pending_models_.front();
    pending_models_.pop_front();
    lk.unlock();
    try {
      model->wait_for_completion();
    } catch (...) {
      lk.lock();
      available_models_.push_back(model);
      throw;
    }
    lk.lock();
    available_models_.push_back(model);
  }
};

} // namespace aot_inductor
} // namespace torch

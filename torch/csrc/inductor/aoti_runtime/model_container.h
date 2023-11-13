#pragma once

#include <algorithm>
#include <deque>
#include <future>
#include <mutex>
#include <shared_mutex>

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#include <torch/csrc/inductor/aoti_runtime/model.h>

namespace torch {
namespace aot_inductor {

class AOTInductorModelContainer {
 public:
  AOTInductorModelContainer(
      size_t num_models,
      bool is_cpu = false,
      std::optional<std::string> cubin_dir = std::nullopt) {
    constants_ = std::make_shared<ConstantMap>();
    models_.reserve(num_models);
    available_models_.reserve(num_models);
    for (size_t i = 0; i < num_models; ++i) {
      models_.push_back(AOTInductorModel::Create(constants_, cubin_dir));
      available_models_.push_back(models_.back().get());
    }

    // Note that the all following fields (input_names_, output_names,
    // etc) can be filled in by the AOT
    // codegen. However, we choose to query such information from
    // the owned AOTInductorModel for a couple of reasons:
    //   * simplify the codegen templates
    //   * reduce information fragmentation and duplication
    //   * the initialization process below is done only once when the container
    //     is constructed, so it would have little performance impact
    auto* model = available_models_[0];
    size_t num_inputs = model->num_inputs();
    input_names_.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; i++) {
      input_names_.push_back(model->input_name(i));
    }

    size_t num_outputs = model->num_outputs();
    output_names_.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; i++) {
      output_names_.push_back(model->output_name(i));
    }

    model->load_constants(is_cpu);
#ifdef USE_CUDA
    constant_blob_ = model->release_constant_blob();
#endif

    for (auto& model : models_) {
      model->update_constants_map(constants_);
    }

    in_spec_ = model->get_in_spec();
    out_spec_ = model->get_out_spec();
  }

  void run(
      AtenTensorHandle*
          input_handles, // array of input AtenTensorHandle; handles
                         // are stolen; the array itself is borrowed
      AtenTensorHandle*
          output_handles, // array for writing output AtenTensorHandle; handles
                          // will be stolen by the caller; the array itself is
                          // borrowed
      DeviceStreamType stream,
      AOTIProxyExecutorHandle proxy_executor) {
    auto* model = get_available_model();
    try {
      model->run(input_handles, output_handles, stream, proxy_executor);
    } catch (...) {
      std::lock_guard lk(models_mutex_);
      available_models_.push_back(model);
      throw;
    }

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

  size_t num_models() const {
    return models_.size();
  }

  const char* get_in_spec() const {
    return in_spec_;
  }

  const char* get_out_spec() const {
    return out_spec_;
  }

 private:
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  const char* in_spec_;
  const char* out_spec_;

#ifdef USE_CUDA
  // Holds the blob storage for constants' at::Tensor for CUDA.
  CUDAPtr constant_blob_;
#endif // USE_CUDA

  // Holds the mapping of constants to at::Tensor.
  // The underlying data of at::Tensor is in either constant_blob_ (for CUDA).
  // or _binary_constants_bin_start (for CPU).
  std::shared_ptr<ConstantMap> constants_;

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

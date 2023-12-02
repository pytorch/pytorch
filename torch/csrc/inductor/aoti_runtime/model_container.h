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
    model_ = AOTInductorModel::Create(constants_, cubin_dir);

    // Note that the all following fields (input_names_, output_names,
    // etc) can be filled in by the AOT
    // codegen. However, we choose to query such information from
    // the owned AOTInductorModel for a couple of reasons:
    //   * simplify the codegen templates
    //   * reduce information fragmentation and duplication
    //   * the initialization process below is done only once when the container
    //     is constructed, so it would have little performance impact
    size_t num_inputs = model_->num_inputs();
    input_names_.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; i++) {
      input_names_.push_back(model_->input_name(i));
    }

    size_t num_outputs = model_->num_outputs();
    output_names_.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; i++) {
      output_names_.push_back(model_->output_name(i));
    }

    model_->load_constants(is_cpu);
#ifdef USE_CUDA
    constant_blob_ = model_->release_constant_blob();
#endif

    model_->update_constants_map(constants_);

    in_spec_ = model_->get_in_spec();
    out_spec_ = model_->get_out_spec();
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
    try {
      model_->run(input_handles, output_handles, stream, proxy_executor);
    } catch (...) {
      throw;
    }
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
    return 1;
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
  std::unique_ptr<AOTInductorModel> model_;

};

} // namespace aot_inductor
} // namespace torch

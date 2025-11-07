#pragma once

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <shared_mutex>

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#include <torch/csrc/inductor/aoti_runtime/model.h>

namespace torch::aot_inductor {
// The state transition is done by:
// (1) NONE state: The default state when created. This state should only exist
// when model_container is created and no constants are being loaded or updated.
// (2) INITIALIZED state: This state get set whenever we load the constants into
// the buffer. This could be done by load_constants or update_constants_buffer.
// (3) FOLDED state: This state should transition from INITIALIZED after
// const_fold is being invoked.
enum class ConstantState : uint8_t { NONE, INITIALIZED, FOLDED, UNKNOWN };

inline std::string toStringConstantState(ConstantState state) {
  switch (state) {
    case ConstantState::NONE:
      return "ConstantState::NONE";
    case ConstantState::INITIALIZED:
      return "ConstantState::INITIALIZED";
    case ConstantState::FOLDED:
      return "ConstantState::FOLDED";
    case ConstantState::UNKNOWN:
      return "ConstantState::UNKNOWN";
    default:
      return "Unknown enum class state for ConstantState";
  }
}

class AOTInductorModelContainer {
 public:
  AOTInductorModelContainer(
      size_t num_models,
      const std::string& device_str,
      const std::optional<std::string>& cubin_dir = std::nullopt) {
    constants_map_ = std::make_shared<ConstantMap>();
    constants_array_ = std::make_shared<std::vector<ConstantHandle>>();

    models_.reserve(num_models);
    available_models_.reserve(num_models);
    for (size_t i = 0; i < num_models; ++i) {
      models_.push_back(AOTInductorModel::Create(
          constants_map_, constants_array_, device_str, cubin_dir));
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
      input_names_.emplace_back(model->input_name(static_cast<int64_t>(i)));
    }

    size_t num_outputs = model->num_outputs();
    output_names_.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; i++) {
      output_names_.emplace_back(model->output_name(static_cast<int64_t>(i)));
    }
    model->load_constants();
    constant_blob_ = model->release_constant_blob();
    constants_internal_offset_.resize(
        model->num_constants() - model->num_folded_constants());
    model->compute_constant_blob(blob_size_, constants_internal_offset_);
    constant_folded_ = ConstantState::INITIALIZED;

    for (auto& model : models_) {
      model->update_constants_map(constants_map_);
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
    std::shared_lock model_lk(model_exec_mutex_);
    auto* model = get_available_model();

    ConstantState& const_folded =
        use_secondary_ ? constant_folded_secondary_ : constant_folded_;
    if (const_folded == ConstantState::INITIALIZED) {
      // At this point, constant is not ready yet. We need to call constant
      // folding before we execute the model. We obtain a unique lock at this
      // point to make sure constant is ready for all.
      model_lk.unlock();
      std::unique_lock constants_folding_lk(model_exec_mutex_);
      // Double locking to make sure constant folding is only ran once.
      if (const_folded == ConstantState::INITIALIZED) {
        auto folded_const_map = model->run_const_fold(
            stream, proxy_executor, /* initialization = */ true);
        update_constant_buffer(
            std::move(folded_const_map),
            /* use_inactive = */ false,
            /* validate_full_update = */ false);
        const_folded = ConstantState::FOLDED;
      }
      constants_folding_lk.unlock();
      model_lk.lock();
    } else if (const_folded != ConstantState::FOLDED) {
      STD_TORCH_CHECK(
          false,
          "Unknown constant state: ",
          toStringConstantState(constant_folded_));
    }

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

  // Non-thread-aware variant of run(). Obviously unsafe to use in a threaded
  // environment :)
  void run_single_threaded(
      AtenTensorHandle*
          input_handles, // array of input AtenTensorHandle; handles
                         // are stolen; the array itself is borrowed
      AtenTensorHandle*
          output_handles, // array for writing output AtenTensorHandle; handles
                          // will be stolen by the caller; the array itself is
                          // borrowed
      DeviceStreamType stream,
      AOTIProxyExecutorHandle proxy_executor) {
    auto* model = available_models_[0];

    ConstantState& const_folded =
        use_secondary_ ? constant_folded_secondary_ : constant_folded_;
    if (const_folded == ConstantState::INITIALIZED) {
      auto folded_const_map = model->run_const_fold(
          stream, proxy_executor, /* initialization = */ true);
      update_constant_buffer(
          std::move(folded_const_map),
          /* use_inactive = */ false,
          /* validate_full_update = */ false);
      const_folded = ConstantState::FOLDED;
    } else if (constant_folded_ != ConstantState::FOLDED) {
      STD_TORCH_CHECK(
          false,
          "Unknown constant state: ",
          toStringConstantState(constant_folded_));
    }

    model->run_single_threaded(
        input_handles, output_handles, stream, proxy_executor);
  }

  const std::unordered_map<std::string, AtenTensorHandle> extract_constants_map(
      bool use_inactive) const {
    size_t n_consts = this->num_constants();
    std::unordered_map<std::string, AtenTensorHandle> ret;
    ret.reserve(n_consts);

    std::shared_ptr<ConstantMap> extract_map = constants_map_;
    // Essentially a XOR
    if (use_inactive != use_secondary_) {
      extract_map = constants_map_secondary_;
    }
    for (size_t idx = 0; idx < n_consts; idx++) {
      if (this->constant_from_folded(idx)) {
        continue;
      }

      auto it = extract_map->find(this->constant_name(idx));
      if (it != extract_map->end()) {
        ret.emplace(this->constant_original_fqn(idx), it->second);
        continue;
      }
    }

    return ret;
  }

  size_t num_constants() const {
    STD_TORCH_CHECK(
        this->num_models() != 0, "No available models in container!");

    return models_[0]->num_constants();
  }

  // retrieve the constant name of constants_info_[idx]
  const char* constant_name(size_t idx) const {
    STD_TORCH_CHECK(
        this->num_models() != 0, "No available models in container!");

    return models_[0]->constant_name(static_cast<int64_t>(idx));
  }

  // retrieve original FQN of constants_info_[idx]
  const char* constant_original_fqn(size_t idx) const {
    STD_TORCH_CHECK(
        this->num_models() != 0, "No available models in container!");

    return models_[0]->constant_original_fqn(static_cast<int64_t>(idx));
  }

  // retrieve whether constant is from folded of constants_info_[idx]
  bool constant_from_folded(size_t idx) const {
    STD_TORCH_CHECK(
        this->num_models() != 0, "No available models in container!");

    return models_[0]->constant_from_folded(static_cast<int64_t>(idx));
  }

  size_t constant_data_size(size_t idx) const {
    STD_TORCH_CHECK(
        this->num_models() != 0, "No available models in container!");

    return models_[0]->constant_data_size(static_cast<int64_t>(idx));
  }

  // retrieve type of constants_info_[idx]
  int32_t constant_type(size_t idx) const {
    STD_TORCH_CHECK(
        this->num_models() != 0, "No available models in container!");

    return models_[0]->constant_type(static_cast<int64_t>(idx));
  }

  // retrieve dtype of constants_info_[idx]
  int32_t constant_dtype(size_t idx) const {
    STD_TORCH_CHECK(
        this->num_models() != 0, "No available models in container!");

    return models_[0]->constant_dtype(static_cast<int64_t>(idx));
  }

  uint64_t constant_blob_size() const {
    if (this->num_models() == 0) {
      throw std::runtime_error("No available models in container!");
    }
    return models_[0]->constant_blob_size();
  }

  void update_constants_from_blob(const uint8_t* weight_blob_ptr) {
    if (this->num_models() == 0) {
      throw std::runtime_error("No available models in container!");
    }
    return models_[0]->update_constants_from_blob(weight_blob_ptr);
  }

  void run_const_fold(
      bool inactive_buffer,
      DeviceStreamType stream,
      AOTIProxyExecutorHandle proxy_executor) {
    AOTInductorModel* model;
    ConstantState& const_folded = inactive_buffer == use_secondary_
        ? constant_folded_
        : constant_folded_secondary_;
    if (!inactive_buffer) {
      // We would need to acquire a unique lock if we want to run constant
      // folding on the active buffer.
      std::unique_lock constants_folding_lk(model_exec_mutex_);
      model = get_available_model();
      try {
        auto folded_const_map = model->run_const_fold(stream, proxy_executor);
        update_constant_buffer(
            std::move(folded_const_map),
            /* use_inactive = */ false,
            /* validate_full_update = */ false);
        const_folded = ConstantState::FOLDED;
      } catch (...) {
        std::lock_guard lk(models_mutex_);
        available_models_.push_back(model);
        throw;
      }
    } else {
      std::shared_lock model_lk(model_exec_mutex_);
      model = get_available_model();

      // We swap the constant mapping to the inactive buffer in the model to run
      // const run.
      auto constants_map = get_constants_map(/* get_inactive= */ true);
      auto constants_array = get_constants_array(/* get_inactive= */ true);

      try {
        model->update_constants_map(
            constants_map, /* remap_constants_array= */ false);
        model->update_constants_array(constants_array);

        auto folded_const_map = model->run_const_fold(stream, proxy_executor);
        update_constant_buffer(
            std::move(folded_const_map),
            /* use_inactive = */ true,
            /* validate_full_update = */ false);

        // Swap back the model's constants mapping
        constants_map = get_constants_map(/* get_inactive= */ false);
        constants_array = get_constants_array(/* get_inactive= */ false);
        model->update_constants_map(
            constants_map, /* remap_constants_array= */ false);
        model->update_constants_array(constants_array);
        const_folded = ConstantState::FOLDED;
      } catch (...) {
        std::lock_guard lk(models_mutex_);
        available_models_.push_back(model);
        throw;
      }
    }

    {
      std::lock_guard lk(models_mutex_);
      pending_models_.push_back(model);
    }
    pending_models_available_.notify_one();
  }

  bool _is_tensor_constant_type(const size_t idx) const {
    auto constant_type = models_[0]->constant_type(static_cast<int64_t>(idx));
    // We should skip constants
    return constant_type == ConstantType::TensorConstant;
  }

  bool _is_buffer_type(const size_t idx) const {
    auto constant_type = models_[0]->constant_type(static_cast<int64_t>(idx));
    // Buffer can be optionally skipped, so if it not provided by upstream
    // services, it is OK to relax the check.
    return constant_type == ConstantType::Buffer;
  }

  bool _is_empty_parameter_type(const size_t idx) const {
    auto constant_type = models_[0]->constant_type(static_cast<int64_t>(idx));
    auto constant_data_size =
        models_[0]->constant_data_size(static_cast<int64_t>(idx));
    // Empty parameters are skipped and not provided by the upstream services,
    // it is OK to skip.
    return constant_type == ConstantType::Parameter && constant_data_size == 0;
  }

  bool _is_tensor_constant_or_buffer_type_or_empty_parameter(
      const size_t idx) const {
    return _is_tensor_constant_type(idx) || _is_buffer_type(idx) ||
        _is_empty_parameter_type(idx);
  }

  void assert_all_constants(
      const std::unordered_map<std::string, AtenTensorHandle>& constants_map) {
    auto num_constants = models_[0]->num_constants();
    for (size_t idx = 0; idx < num_constants; idx++) {
      if (models_[0]->constant_from_folded(static_cast<int64_t>(idx))) {
        continue;
      }

      auto constant_name =
          std::string(models_[0]->constant_name(static_cast<int64_t>(idx)));
      auto it = constants_map.find(constant_name);
      if (it == constants_map.end()) {
        if (_is_tensor_constant_or_buffer_type_or_empty_parameter(idx)) {
          // tracing sometimes creates tensors that are non-existent in
          // original graph. We could skip those and do a direct copy.
          std::cerr << "[WARNING] Found constant or module state buffer or "
                    << "empty module state parameter " << constant_name
                    << " in model, but not provided by user!\n";
          continue;
        }

        STD_TORCH_CHECK(
            false,
            "Cannot find constants ",
            constant_name,
            " in constants_map!");
      }
    }
  }

  // We directly take ownership from AtenTensorHandle if constants are moved.
  void update_constant_buffer(
      std::unordered_map<std::string, AtenTensorHandle>&& constants_map,
      bool use_inactive,
      bool validate_full_update) {
    STD_TORCH_CHECK(
        this->num_models() != 0, "No available models in container!");
    if (validate_full_update) {
      assert_all_constants(constants_map);
    }

    ConstantState& const_folded = use_inactive == use_secondary_
        ? constant_folded_
        : constant_folded_secondary_;
    const_folded = ConstantState::INITIALIZED;

    auto original_constants_map = get_constants_map(!use_inactive);
    auto constants_map_to_update = get_constants_map(use_inactive);

    auto num_constants = models_[0]->num_constants();
    for (size_t idx = 0; idx < num_constants; idx++) {
      auto constant_name =
          std::string(models_[0]->constant_name(static_cast<int64_t>(idx)));
      auto it = constants_map.find(constant_name);
      if (it == constants_map.end() &&
          !(use_inactive && _is_tensor_constant_type(idx))) {
        continue;
      }

      AtenTensorHandle tensor;
      if (it == constants_map.end()) {
        aoti_torch_clone(
            original_constants_map->find(constant_name)->second.get(), &tensor);
      } else {
        tensor = it->second;
      }

      constants_map_to_update->insert_or_assign(
          constant_name, RAIIAtenTensorHandle(tensor));
    }
    // Update the inactive constant array.
    update_array_from_map(
        get_constants_array(use_inactive), constants_map_to_update);
  }

  // This function updates the buffer for storing constants.
  // It will update the buffer, the mapping and the array mapping.
  void update_constant_buffer(
      const std::unordered_map<std::string, AtenTensorHandle>& constants_map,
      bool use_inactive,
      bool validate_full_update,
      bool user_managed = false) {
    STD_TORCH_CHECK(
        this->num_models() != 0, "No model available in container!");

    if (validate_full_update) {
      assert_all_constants(constants_map);
    }

    ConstantState& const_folded = use_inactive == use_secondary_
        ? constant_folded_
        : constant_folded_secondary_;
    const_folded = ConstantState::INITIALIZED;

    auto original_constants_map = get_constants_map(!use_inactive);
    auto constants_map_to_update = get_constants_map(use_inactive);

    auto num_constants = models_[0]->num_constants();
    for (size_t idx = 0; idx < num_constants; idx++) {
      auto constant_name =
          std::string(models_[0]->constant_name(static_cast<int64_t>(idx)));
      auto it = constants_map.find(constant_name);
      if (it == constants_map.end() &&
          !(use_inactive &&
            _is_tensor_constant_or_buffer_type_or_empty_parameter(idx))) {
        continue;
      }

      AtenTensorHandle tensor;
      if (it == constants_map.end()) {
        tensor = original_constants_map->find(constant_name)->second.get();
      } else {
        tensor = it->second;
      }

      if (user_managed) {
        // If user managed, we pass in the pointer directly, and skip the
        // copy.
        constants_map_to_update->insert_or_assign(
            constant_name,
            MaybeOwningAtenTensorHandle(tensor, /* user_managed = */ true));
        continue;
      }

      auto* constants_blob_ptr =
          static_cast<uint8_t*>(get_constant_blob_ptr(use_inactive));

      // Move the data to container handled blob.
      uint8_t* internal_constants_ptr =
          constants_blob_ptr + constants_internal_offset_[idx];
      void* user_constant_ptr;
      int64_t constant_size;
      int64_t* stride;
      int64_t offset;
      aoti_torch_get_data_ptr(tensor, &user_constant_ptr);
      aoti_torch_get_storage_size(tensor, &constant_size);
      AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(tensor, &stride));
      AOTI_TORCH_ERROR_CODE_CHECK(
          aoti_torch_get_storage_offset(tensor, &offset));
      auto dtype = models_[0]->constant_dtype(idx);

#ifdef USE_XPU
      sycl::queue* queue_ptr = nullptr;
      aoti_torch_get_current_sycl_queue((void**)&queue_ptr);
      queue_ptr
          ->memcpy(internal_constants_ptr, user_constant_ptr, constant_size)
          .wait();
#elif USE_MPS
      internal_constants_ptr = constants_blob_ptr;
      aoti_torch_mps_copy_buffer(
          user_constant_ptr,
          constants_blob_ptr,
          constant_size,
          offset,
          constants_internal_offset_[idx]);
      // For mps tensors, all constants are stored in one buffer, with the
      // offset being where the constant starts. So we want to change the
      // constant tensor's offset to point to constants_internal_offset_[idx]
      offset = constants_internal_offset_[idx] /
          aoti_torch_dtype_element_size(dtype);
#elif USE_CUDA
      AOTI_RUNTIME_CUDA_CHECK(cudaMemcpy(
          internal_constants_ptr,
          user_constant_ptr,
          constant_size,
          cudaMemcpyDefault));
#else
      memcpy(internal_constants_ptr, user_constant_ptr, constant_size);
#endif
      // Generate Tensor from container handled blob.
      // We extract stride and offset from provided Tensor since we do not
      // guarantee that the tensor is contiguous.
      AtenTensorHandle tensor_handle;
      int device_type = models_[0]->get_device_type();
      int device_idx = models_[0]->get_device_idx();
      AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_create_tensor_from_blob(
          internal_constants_ptr,
          models_[0]->constant_ndim(idx),
          models_[0]->constant_shape(idx),
          stride,
          offset,
          dtype,
          device_type,
          device_idx,
          &tensor_handle));

      // Now place the tensor to constants_map. Note at this point the
      // ownership of the tensor_handle will be taken over.
      constants_map_to_update->insert_or_assign(
          constant_name, RAIIAtenTensorHandle(tensor_handle));
    }
    // Update the inactive constant array.
    update_array_from_map(
        get_constants_array(use_inactive), constants_map_to_update);
  }

  void update_array_from_map(
      const std::shared_ptr<std::vector<ConstantHandle>>& constants_array,
      const std::shared_ptr<ConstantMap>& constants_map) {
    auto num_constants = models_[0]->num_constants();
    for (size_t idx = 0; idx < num_constants; idx++) {
      if (constants_map->find(models_[0]->constant_name(
              static_cast<int64_t>(idx))) != constants_map->end()) {
        constants_array->at(idx) = ConstantHandle(
            constants_map
                ->find(models_[0]->constant_name(static_cast<int64_t>(idx)))
                ->second);
      }
    }
  }

  void swap_constant_buffer() {
    std::lock_guard unique_lk(model_exec_mutex_);

    auto constants_map = get_constants_map(/* get_inactive= */ true);
    auto constants_array = get_constants_array(/* get_inactive= */ true);

    for (auto& model : models_) {
      model->update_constants_map(
          constants_map, /* remap_constants_array = */ false);
      model->update_constants_array(constants_array);
    }

    use_secondary_ = !use_secondary_;
  }

  void free_inactive_constant_buffer() {
    if (use_secondary_) {
      constant_folded_ = ConstantState::NONE;
      constant_blob_.reset();
    } else {
      constant_folded_secondary_ = ConstantState::NONE;
      constant_blob_secondary_.reset();
    }
    // Free the internally held constants
    int num_constants = static_cast<int>(models_[0]->num_constants());
    std::shared_ptr<ConstantMap> to_free_map =
        use_secondary_ ? constants_map_ : constants_map_secondary_;

    for (int i = 0; i < num_constants; i++) {
      if (models_[0]->constant_from_folded(i)) {
        auto it = to_free_map->find(models_[0]->constant_name(i));
        if (it != to_free_map->end()) {
          it->second.reset();
        }
      }
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

  // Holds the blob storage for constants' at::Tensor within the container.
  // This blob of memory will be managed by the container.
  RAIIDataPtr constant_blob_;
  RAIIDataPtr constant_blob_secondary_;

  size_t blob_size_;
  std::vector<size_t> constants_internal_offset_;

  // Determine which constants is being used for the model.
  // If true,
  // constants_map_secondary/constant_blob_secondary/constants_array_secondary
  // is being used.
  bool use_secondary_{false};

  // Determine whether we have ran constant folding
  ConstantState constant_folded_{ConstantState::NONE};
  ConstantState constant_folded_secondary_{ConstantState::NONE};

  // Holds the mapping of constants to at::Tensor.
  // The underlying data of at::Tensor is in either constant_blob_ (for CUDA).
  // or _binary_constants_bin_start (for CPU).
  std::shared_ptr<ConstantMap> constants_map_;
  std::shared_ptr<ConstantMap> constants_map_secondary_;

  // Holds the indexed array of constant for faster lookup during runtime.
  std::shared_ptr<std::vector<ConstantHandle>> constants_array_;
  std::shared_ptr<std::vector<ConstantHandle>> constants_array_secondary_;

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

  // This mutex is used to protect execution of model.
  // We acquire the mutex in shared mode if we allow concurrent execution.
  // We acquire the mutex in unique mode when we want exclusive access of the
  // model. One such case is when we want to do a weight swapping. We want to
  // make sure no one is executing the model.
  std::shared_mutex model_exec_mutex_;

  RAIIDataPtr allocate_constant_blob() {
#if defined(USE_CUDA) || defined(USE_XPU) || defined(USE_MPS)
    return RAII_gpuMalloc(blob_size_);
#else
    return RAII_cpuMalloc(blob_size_);
#endif // USE_CUDA
  }

  void* get_constant_blob_ptr(bool get_inactive) {
    if ((get_inactive && use_secondary_) ||
        (!get_inactive && !use_secondary_)) {
      if (!constant_blob_) {
        constant_blob_ = allocate_constant_blob();
      }
      return constant_blob_.get();
    } else {
      if (!constant_blob_secondary_) {
        constant_blob_secondary_ = allocate_constant_blob();
      }
      return constant_blob_secondary_.get();
    }
  }

  std::shared_ptr<ConstantMap> get_constants_map(bool get_inactive) {
    if ((get_inactive && use_secondary_) ||
        (!get_inactive && !use_secondary_)) {
      return constants_map_;
    } else {
      if (!constants_map_secondary_) {
        constants_map_secondary_ = std::make_shared<ConstantMap>();
      }
      return constants_map_secondary_;
    }
  }

  std::shared_ptr<std::vector<ConstantHandle>> get_constants_array(
      bool get_inactive) {
    if ((get_inactive && use_secondary_) ||
        (!get_inactive && !use_secondary_)) {
      return constants_array_;
    } else {
      if (!constants_array_secondary_) {
        constants_array_secondary_ =
            std::make_shared<std::vector<ConstantHandle>>(
                models_[0]->num_constants());
      }
      return constants_array_secondary_;
    }
  }

  void reclaim_finished_models(std::unique_lock<std::mutex>& lk) {
#ifdef __aarch64__
    // push finished model instances to the end of pending_models_
    auto it = std::partition(
        pending_models_.begin(),
        pending_models_.end(),
        [](AOTInductorModel* m) { return !m->is_finished(); });
#else
    // push finished model instances to the end of pending_models_
    auto it = std::stable_partition(
        pending_models_.begin(),
        pending_models_.end(),
        [](AOTInductorModel* m) { return !m->is_finished(); });
#endif

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

} // namespace torch::aot_inductor

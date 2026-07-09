#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <shared_mutex>

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#include <torch/csrc/inductor/aoti_runtime/interface.h>
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

struct ConstantBufferSet {
  RAIIDataPtr blob;
  RAIIDataPtr aux_cpu_blob;
  std::shared_ptr<ConstantMap> map;
  std::shared_ptr<std::vector<ConstantHandle>> array;
  ConstantState fold_state{ConstantState::NONE};

  void* ensure_blob(size_t blob_size) {
    if (!blob) {
#if defined(USE_CUDA) || defined(USE_XPU) || defined(USE_MPS)
      blob = RAII_gpuMalloc(blob_size);
#else
      blob = RAII_cpuMalloc(blob_size);
#endif
    }
    return blob.get();
  }

  void* ensure_aux_cpu_blob(size_t aux_cpu_blob_size) {
    if (!aux_cpu_blob) {
      aux_cpu_blob = RAII_cpuMalloc(aux_cpu_blob_size);
    }
    return aux_cpu_blob.get();
  }

  void update_array(AOTInductorModel* model) {
    auto num_constants = model->num_constants();
    for (size_t idx = 0; idx < num_constants; idx++) {
      auto it = map->find(model->constant_name(static_cast<int64_t>(idx)));
      if (it != map->end()) {
        array->at(idx) = ConstantHandle(it->second);
      }
    }
  }

  void reset(AOTInductorModel* model) {
    fold_state = ConstantState::NONE;
    blob.reset();
    aux_cpu_blob.reset();
    int num_constants = static_cast<int>(model->num_constants());
    for (int i = 0; i < num_constants; i++) {
      if (model->constant_from_folded(i)) {
        auto it = map->find(model->constant_name(i));
        if (it != map->end()) {
          it->second.reset();
        }
      }
    }
  }
};

class AOTInductorModelContainer {
 public:
  AOTInductorModelContainer(
      size_t num_models,
      const std::string& device_str,
      const std::optional<std::string>& cubin_dir = std::nullopt) {
    buffers_[0].map = std::make_shared<ConstantMap>();
    buffers_[0].array = std::make_shared<std::vector<ConstantHandle>>();

    models_.reserve(num_models);
    available_models_.reserve(num_models);
    for (size_t i = 0; i < num_models; ++i) {
      models_.push_back(AOTInductorModel::Create(
          buffers_[0].map, buffers_[0].array, device_str, cubin_dir));
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
    buffers_[0].blob = model->release_constant_blob();
    buffers_[0].aux_cpu_blob = model->release_aux_cpu_constant_blob();
    constants_internal_offset_.resize(
        model->num_constants() - model->num_folded_constants());
    aux_cpu_constants_internal_offset_.resize(
        model->num_constants() - model->num_folded_constants());
    model->compute_constant_blob(
        blob_size_,
        constants_internal_offset_,
        aux_cpu_blob_size_,
        aux_cpu_constants_internal_offset_);
    buffers_[0].fold_state = ConstantState::INITIALIZED;

    for (auto& m : models_) {
      m->update_constants_map(buffers_[0].map);
    }

    buffers_[1].map = std::make_shared<ConstantMap>();
    buffers_[1].array =
        std::make_shared<std::vector<ConstantHandle>>(model->num_constants());

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

    auto& const_folded = active().fold_state;
    if (const_folded == ConstantState::INITIALIZED) {
      // Do NOT call get_available_model() before upgrading to exclusive lock.
      // Holding a model across the upgrade causes a deadlock when another
      // thread holds a shared lock and waits for the model.
      model_lk.unlock();
      std::unique_lock constants_folding_lk(model_exec_mutex_);
      // Double locking to make sure constant folding is only ran once.
      if (const_folded == ConstantState::INITIALIZED) {
        auto* model = get_available_model();
        // TODO: add try catch block to handle exception.
        auto folded_const_map = model->run_const_fold(
            stream, proxy_executor, /* initialization = */ true);
        update_constant_buffer(
            std::move(folded_const_map),
            /* use_inactive = */ false,
            /* validate_full_update = */ false);
        const_folded = ConstantState::FOLDED;
        {
          std::lock_guard lk(models_mutex_);
          pending_models_.push_back(model);
        }
        pending_models_available_.notify_one();
      }
      constants_folding_lk.unlock();
      model_lk.lock();
    } else if (const_folded != ConstantState::FOLDED) {
      throw std::runtime_error(
          "Unknown constant state: " + toStringConstantState(const_folded));
    }

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

    auto& const_folded = active().fold_state;
    if (const_folded == ConstantState::INITIALIZED) {
      auto folded_const_map = model->run_const_fold(
          stream, proxy_executor, /* initialization = */ true);
      update_constant_buffer(
          std::move(folded_const_map),
          /* use_inactive = */ false,
          /* validate_full_update = */ false);
      const_folded = ConstantState::FOLDED;
    } else if (const_folded != ConstantState::FOLDED) {
      throw std::runtime_error(
          "Unknown constant state: " + toStringConstantState(const_folded));
    }

    model->run_single_threaded(
        input_handles, output_handles, stream, proxy_executor);
  }

  const std::unordered_map<std::string, AtenTensorHandle> extract_constants_map(
      bool use_inactive) const {
    size_t n_consts = this->num_constants();
    std::unordered_map<std::string, AtenTensorHandle> ret;
    ret.reserve(n_consts);

    const auto& extract_map = use_inactive ? inactive().map : active().map;
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

  const std::vector<AOTInductorConstantMapEntry>& extract_constants_map_entries(
      bool use_inactive) {
    size_t n_consts = this->num_constants();
    extracted_constant_map_entry_names_.clear();
    extracted_constant_map_entries_.clear();
    extracted_constant_map_entry_names_.reserve(n_consts);
    extracted_constant_map_entries_.reserve(n_consts);

    const auto& extract_map = use_inactive ? inactive().map : active().map;
    for (size_t idx = 0; idx < n_consts; idx++) {
      if (this->constant_from_folded(idx)) {
        continue;
      }

      auto it = extract_map->find(this->constant_name(idx));
      if (it != extract_map->end()) {
        extracted_constant_map_entry_names_.emplace_back(
            this->constant_original_fqn(idx));
        extracted_constant_map_entries_.push_back(
            {extracted_constant_map_entry_names_.back().c_str(), it->second});
      }
    }

    return extracted_constant_map_entries_;
  }

  size_t num_constants() const {
    if (this->num_models() == 0) {
      throw std::runtime_error("No available models in container!");
    }
    return models_[0]->num_constants();
  }

  // retrieve the constant name of constants_info_[idx]
  const char* constant_name(size_t idx) const {
    if (this->num_models() == 0) {
      throw std::runtime_error("No available models in container!");
    }
    return models_[0]->constant_name(static_cast<int64_t>(idx));
  }

  // retrieve original FQN of constants_info_[idx]
  const char* constant_original_fqn(size_t idx) const {
    if (this->num_models() == 0) {
      throw std::runtime_error("No available models in container!");
    }
    return models_[0]->constant_original_fqn(static_cast<int64_t>(idx));
  }

  // retrieve whether constant is from folded of constants_info_[idx]
  bool constant_from_folded(size_t idx) const {
    if (this->num_models() == 0) {
      throw std::runtime_error("No available models in container!");
    }
    return models_[0]->constant_from_folded(static_cast<int64_t>(idx));
  }

  size_t constant_data_size(size_t idx) const {
    if (this->num_models() == 0) {
      throw std::runtime_error("No available models in container!");
    }
    return models_[0]->constant_data_size(static_cast<int64_t>(idx));
  }

  // retrieve type of constants_info_[idx]
  int32_t constant_type(size_t idx) const {
    if (this->num_models() == 0) {
      throw std::runtime_error("No available models in container!");
    }
    return models_[0]->constant_type(static_cast<int64_t>(idx));
  }

  // retrieve dtype of constants_info_[idx]
  int32_t constant_dtype(size_t idx) const {
    if (this->num_models() == 0) {
      throw std::runtime_error("No available models in container!");
    }
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
    auto& const_folded =
        inactive_buffer ? inactive().fold_state : active().fold_state;
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
      auto inactive_map = inactive().map;
      auto inactive_array = inactive().array;

      try {
        model->update_constants_map(
            inactive_map, /* remap_constants_array= */ false);
        model->update_constants_array(inactive_array);

        auto folded_const_map = model->run_const_fold(stream, proxy_executor);
        update_constant_buffer(
            std::move(folded_const_map),
            /* use_inactive = */ true,
            /* validate_full_update = */ false);

        // Swap back the model's constants mapping
        auto active_map = active().map;
        auto active_array = active().array;
        model->update_constants_map(
            active_map, /* remap_constants_array= */ false);
        model->update_constants_array(active_array);
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
        throw std::runtime_error(
            std::string("Cannot find constants ") + constant_name +
            std::string(" in constants_map!"));
      }
    }
  }

  // We directly take ownership from AtenTensorHandle if constants are moved.
  void update_constant_buffer(
      std::unordered_map<std::string, AtenTensorHandle>&& constants_map,
      bool use_inactive,
      bool validate_full_update) {
    if (this->num_models() == 0) {
      throw std::runtime_error("No model available in container!");
    }
    if (validate_full_update) {
      assert_all_constants(constants_map);
    }

    auto& target = use_inactive ? inactive() : active();
    auto& source = use_inactive ? active() : inactive();
    target.fold_state = ConstantState::INITIALIZED;

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
            source.map->find(constant_name)->second.get(), &tensor);
      } else {
        tensor = it->second;
      }

      target.map->insert_or_assign(constant_name, RAIIAtenTensorHandle(tensor));
    }
    target.update_array(models_[0].get());
  }

  // This function updates the buffer for storing constants.
  // It will update the buffer, the mapping and the array mapping.
  // When allow_h2d_copy is true, CPU input tensors are silently copied to the
  // model's device (via the same memcpy path used for same-device copies).
  // Note: allow_h2d_copy is incompatible with user_managed, since user_managed
  // mode stores the tensor pointer directly rather than copying.
  void update_constant_buffer(
      const std::unordered_map<std::string, AtenTensorHandle>& constants_map,
      bool use_inactive,
      bool validate_full_update,
      bool user_managed = false,
      bool allow_h2d_copy = false) {
    if (this->num_models() == 0) {
      throw std::runtime_error("No model available in container!");
    }
    if (validate_full_update) {
      assert_all_constants(constants_map);
    }
    if (allow_h2d_copy && user_managed) {
      throw std::runtime_error(
          "update_constant_buffer: allow_h2d_copy is not supported with user_managed");
    }

    int32_t cpu_device_type = aoti_torch_device_type_cpu();
    auto num_constants = models_[0]->num_constants();
    for (size_t idx = 0; idx < num_constants; idx++) {
      if (models_[0]->constant_from_folded(static_cast<int64_t>(idx))) {
        continue;
      }
      auto constant_name =
          std::string(models_[0]->constant_name(static_cast<int64_t>(idx)));
      auto it = constants_map.find(constant_name);
      if (it == constants_map.end()) {
        continue;
      }
      int32_t expected_const_device_type =
          models_[0]->constant_device_type(static_cast<int64_t>(idx));
      int32_t tensor_device_type = 0;
      AOTI_TORCH_ERROR_CODE_CHECK(
          aoti_torch_get_device_type(it->second, &tensor_device_type));
      if (tensor_device_type != expected_const_device_type) {
#ifndef USE_MPS
        if (allow_h2d_copy && tensor_device_type == cpu_device_type) {
          continue;
        }
#endif
        throw std::runtime_error(
            "update_constant_buffer: constant '" + constant_name +
            "' is on device type " + std::to_string(tensor_device_type) +
            " but expected device type " +
            std::to_string(expected_const_device_type));
      }
    }

    int32_t model_device_type = models_[0]->get_device_type();

    auto& target = use_inactive ? inactive() : active();
    auto& source = use_inactive ? active() : inactive();
    target.fold_state = ConstantState::INITIALIZED;

    // Running indices into constants_internal_offset_ and
    // aux_cpu_constants_internal_offset_, which hold per-blob offsets
    // only for non-folded constants (mirrors compute_constant_blob's
    // bookkeeping in model_base.h). Advance per-blob for every non-folded
    // constant as we walk.
    size_t main_blob_idx = 0;
    size_t aux_cpu_blob_idx = 0;
#ifdef USE_CUDA
    // Opt-in pinned async staging pool for the delta-update H2D copies below.
    // nullptr (default / on allocation failure) keeps the throttled path.
    auto staging_pool = tryMakeConstantsStagingPool();
#endif
    auto _update_start = std::chrono::steady_clock::now();
    AOTI_LOG_LOADING(
        "update_constant_buffer: starting copy of " << num_constants
                                                    << " constants");
    for (size_t idx = 0; idx < num_constants; idx++) {
      if (models_[0]->constant_from_folded(static_cast<int64_t>(idx))) {
        continue;
      }
      int32_t const_device_type =
          models_[0]->constant_device_type(static_cast<int64_t>(idx));
      bool is_aux_cpu = const_device_type != model_device_type;

      size_t this_main_idx = main_blob_idx;
      size_t this_aux_cpu_idx = aux_cpu_blob_idx;
      if (is_aux_cpu) {
        aux_cpu_blob_idx++;
      } else {
        main_blob_idx++;
      }

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
        tensor = source.map->find(constant_name)->second.get();
      } else {
        tensor = it->second;
      }

      if (user_managed) {
        // If user managed, we pass in the pointer directly, and skip the
        // copy.
        target.map->insert_or_assign(
            constant_name,
            MaybeOwningAtenTensorHandle(tensor, /* user_managed = */ true));
        continue;
      }

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

      AtenTensorHandle tensor_handle = nullptr;

      if (is_aux_cpu) {
        // CPU constant in a mixed-device model. Write into the container's
        // auxiliary CPU blob at the pre-computed offset, mirroring how the
        // primary blob is managed.
        auto* aux_blob_ptr = static_cast<uint8_t*>(
            target.ensure_aux_cpu_blob(aux_cpu_blob_size_));
        uint8_t* internal_constants_ptr =
            aux_blob_ptr + aux_cpu_constants_internal_offset_[this_aux_cpu_idx];
        memcpy(internal_constants_ptr, user_constant_ptr, constant_size);
        AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_create_tensor_from_blob(
            internal_constants_ptr,
            models_[0]->constant_ndim(idx),
            models_[0]->constant_shape(idx),
            stride,
            offset,
            dtype,
            cpu_device_type,
            /* device_index = */ 0,
            &tensor_handle));
      } else {
        auto* constants_blob_ptr =
            static_cast<uint8_t*>(target.ensure_blob(blob_size_));

        // Move the data to container handled blob.
        uint8_t* internal_constants_ptr =
            constants_blob_ptr + constants_internal_offset_[this_main_idx];

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
            constants_internal_offset_[this_main_idx]);
        // For mps tensors, all constants are stored in one buffer, with the
        // offset being where the constant starts. So we want to change the
        // constant tensor's offset to point to
        // constants_internal_offset_[this_main_idx]
        offset = constants_internal_offset_[this_main_idx] /
            aoti_torch_dtype_element_size(dtype);
#elif USE_CUDA
        // Pinned-async staging only helps for pageable host sources (which
        // trigger CUDA/HIP's device-wide implicit sync) and is only safe for
        // them, since copyH2DViaStage CPU-reads the source. Device / pinned
        // sources use the synchronous throttled copy, which serializes with
        // prior device work on the source; the pool's private stream would not.
        // Detect per constant: a single update may mix pageable-host and
        // device/pinned sources, so the type cannot be cached across the loop.
        bool use_staging = false;
        if (staging_pool != nullptr) {
          cudaPointerAttributes attrs{};
          cudaError_t pointer_attrs_result =
              cudaPointerGetAttributes(&attrs, user_constant_ptr);
          if (pointer_attrs_result != cudaSuccess) {
            (void)cudaGetLastError();
            use_staging = true;
          } else {
            use_staging = (attrs.type == cudaMemoryTypeUnregistered);
          }
        }
        if (use_staging) {
          staging_pool->copyH2DViaStage(
              internal_constants_ptr,
              user_constant_ptr,
              static_cast<size_t>(constant_size));
        } else {
          aoti_cuda_memcpy_throttled(
              internal_constants_ptr,
              user_constant_ptr,
              static_cast<size_t>(constant_size),
              cudaMemcpyDefault);
        }
#else
        memcpy(internal_constants_ptr, user_constant_ptr, constant_size);
#endif
        // Generate Tensor from container handled blob.
        // We extract stride and offset from provided Tensor since we do not
        // guarantee that the tensor is contiguous.
        int device_idx = models_[0]->get_device_idx();
        AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_create_tensor_from_blob(
            internal_constants_ptr,
            models_[0]->constant_ndim(idx),
            models_[0]->constant_shape(idx),
            stride,
            offset,
            dtype,
            model_device_type,
            device_idx,
            &tensor_handle));
      }

      // Now place the tensor to constants_map. Note at this point the
      // ownership of the tensor_handle will be taken over.
      target.map->insert_or_assign(
          constant_name, RAIIAtenTensorHandle(tensor_handle));
    }
#ifdef USE_CUDA
    // Synchronize the staging stream (surfacing any async copy error) and
    // release the pinned buffers before the updated constants are observed by
    // callers.
    if (staging_pool != nullptr) {
      staging_pool->finish();
    }
    staging_pool.reset();
#endif
    auto _update_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - _update_start)
                          .count();
    AOTI_LOG_LOADING(
        "update_constant_buffer: copy completed in " << _update_ms << " ms");
    target.update_array(models_[0].get());

#ifdef USE_CUDA
    // S638065: the GPU constant copies above run on the default stream (raw
    // cudaMemcpy), while run_const_fold launches the fold on
    // getCurrentCUDAStream(). On ROCm the default stream is not implicitly
    // ordered with the fold's stream, so without this the fold could read
    // not-yet-copied weights and bake stale values into the folded constants.
    // Wait for the copies to complete before returning, so any subsequent fold
    // (on any stream) sees the updated weights.
    AOTI_RUNTIME_CUDA_CHECK(cudaStreamSynchronize(0));
#endif // USE_CUDA
  }

  void swap_constant_buffer() {
    std::lock_guard unique_lk(model_exec_mutex_);

    active_idx_ = 1 - active_idx_;

    for (auto& model : models_) {
      model->update_constants_map(
          active().map, /* remap_constants_array = */ false);
      model->update_constants_array(active().array);
    }
  }

  void free_inactive_constant_buffer() {
    inactive().reset(models_[0].get());
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

  std::array<ConstantBufferSet, 2> buffers_;
  int active_idx_{0};

  size_t blob_size_;
  std::vector<size_t> constants_internal_offset_;
  size_t aux_cpu_blob_size_;
  std::vector<size_t> aux_cpu_constants_internal_offset_;

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

  std::vector<std::string> extracted_constant_map_entry_names_;
  std::vector<AOTInductorConstantMapEntry> extracted_constant_map_entries_;

  ConstantBufferSet& active() {
    return buffers_[active_idx_];
  }
  ConstantBufferSet& inactive() {
    return buffers_[1 - active_idx_];
  }
  const ConstantBufferSet& active() const {
    return buffers_[active_idx_];
  }
  const ConstantBufferSet& inactive() const {
    return buffers_[1 - active_idx_];
  }

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

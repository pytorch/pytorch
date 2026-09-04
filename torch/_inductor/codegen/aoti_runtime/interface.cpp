// Definition of AOTI runtime interface functions

#include <torch/csrc/inductor/aoti_runtime/interface.h>
#include <torch/csrc/inductor/aoti_runtime/model_container.h>

#include <exception>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Stores the last error message from a failed AOTI runtime call so that
// callers on the other side of the C ABI boundary can retrieve it via
// AOTInductorGetLastError(). Without this, exception messages (e.g.
// "CUDA error: an illegal memory access was encountered") are lost when
// the runtime boundary catches them and returns an error code.
static thread_local std::string g_aoti_last_error;

namespace {

// Error reporting must not let a secondary exception cross the C ABI.
AOTIRuntimeError record_aoti_runtime_exception(
    const std::exception& e) noexcept {
  try {
    g_aoti_last_error = e.what();
  } catch (...) { // NOLINT(bugprone-empty-catch)
  }
  try {
    std::cerr << "Error: " << e.what() << '\n';
  } catch (...) { // NOLINT(bugprone-empty-catch)
  }
  return AOTI_RUNTIME_FAILURE;
}

AOTIRuntimeError record_unknown_aoti_runtime_exception() noexcept {
  try {
    g_aoti_last_error = "Unknown exception";
  } catch (...) { // NOLINT(bugprone-empty-catch)
  }
  try {
    std::cerr << "Unknown exception occurred.\n";
  } catch (...) { // NOLINT(bugprone-empty-catch)
  }
  return AOTI_RUNTIME_FAILURE;
}

} // namespace

#define AOTI_RUNTIME_TRY(...)                       \
  try {                                             \
    g_aoti_last_error.clear();                      \
    __VA_ARGS__                                     \
  } catch (const std::exception& e) {               \
    return record_aoti_runtime_exception(e);        \
  } catch (...) {                                   \
    return record_unknown_aoti_runtime_exception(); \
  }

#define AOTI_VECTOR_SIZE_CHECK(actual_size, expected_size, name)  \
  do {                                                            \
    AOTI_RUNTIME_CHECK(                                           \
        actual_size == expected_size,                             \
        "expected " + std::string(name) + " vector size to be " + \
            std::to_string(expected_size) + ", but got " +        \
            std::to_string(actual_size));                         \
  } while (0)

// AOTInductor uses at::addmm_out, which doesn't support
// arguments that require gradient. For this reason, we
// enforce no_grad context for run APIs.
//
// A RAII, thread local (!) guard that enables or disables grad mode upon
// construction, and sets it back to the original value upon destruction.
struct AOTINoGradGuard {
  AOTINoGradGuard() {
    aoti_torch_grad_mode_set_enabled(false);
  }
  AOTINoGradGuard(const AOTINoGradGuard&) = delete;
  AOTINoGradGuard(AOTINoGradGuard&&) noexcept = delete;
  ~AOTINoGradGuard() {
    aoti_torch_grad_mode_set_enabled(prev_mode);
  }
  AOTINoGradGuard& operator=(const AOTINoGradGuard&) = delete;
  AOTINoGradGuard& operator=(AOTINoGradGuard&&) noexcept = delete;
  bool prev_mode{aoti_torch_grad_mode_is_enabled()};
};

namespace {

std::unordered_map<std::string, AtenTensorHandle> constant_map_from_pairs(
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs) {
  std::unordered_map<std::string, AtenTensorHandle> input_map;
  input_map.reserve(num_pairs);
  for (size_t i = 0; i < num_pairs; ++i) {
    input_map.emplace(pairs[i].name, pairs[i].handle);
  }
  return input_map;
}

// Shared constructor for AOTInductorModelCreate / AOTInductorModelCreateV2.
// `populate(constant_map)` is called between model construction and
// optional embedded-blob loading.
template <typename Populate>
AOTIRuntimeError createModelImpl(
    AOTInductorModelHandle* model_handle,
    bool load_constants_from_blob,
    Populate&& populate) {
  auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
  auto constant_array = std::make_shared<
      std::vector<torch::aot_inductor::ConstantHandle>>();
  auto model = std::make_unique<torch::aot_inductor::AOTInductorModel>(
      constant_map,
      constant_array,
      // device_str is hardcoded, as AOTInductorModelCreate is only used
      // for CPU models.
      "cpu",
      "");
  std::forward<Populate>(populate)(*constant_map);
  if (load_constants_from_blob) {
    model->load_constants();
  }
  *model_handle = reinterpret_cast<AOTInductorModelHandle>(model.release());
  return AOTI_RUNTIME_SUCCESS;
}

} // namespace

extern "C" {

AOTIRuntimeError AOTInductorModelContainerCreate(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    bool is_cpu,
    const char* cubin_dir) AOTI_RUNTIME_TRY({
      return AOTInductorModelContainerCreateWithDevice(
        container_handle,
        num_models,
        is_cpu ? "cpu" : "cuda",
        cubin_dir);
})

AOTIRuntimeError AOTInductorModelContainerCreateWithDevice(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir) AOTI_RUNTIME_TRY({
  if (num_models == 0) {
    std::cerr << "Error: num_models must be positive, but got 0\n";
    return AOTI_RUNTIME_FAILURE;
  }
  std::optional<std::string> cubin_dir_opt;
  if (cubin_dir != nullptr) {
    cubin_dir_opt.emplace(cubin_dir);
  }
  auto* container = new torch::aot_inductor::AOTInductorModelContainer(
      num_models, std::string(device_str), cubin_dir_opt);
  *container_handle =
      reinterpret_cast<AOTInductorModelContainerHandle>(container);
  return AOTI_RUNTIME_SUCCESS;
})


AOTIRuntimeError AOTInductorModelContainerCreateWithExternalConstants(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir,
    const AOTInductorConstantMapEntry* constant_entries,
    size_t num_constant_entries) AOTI_RUNTIME_TRY({
  if (num_models == 0) {
    std::cerr << "Error: num_models must be positive, but got 0\n";
    return AOTI_RUNTIME_FAILURE;
  }
  if (num_constant_entries != 0 && constant_entries == nullptr) {
    std::cerr << "Error: constant_entries is null but num_constant_entries is "
              << num_constant_entries << "\n";
    return AOTI_RUNTIME_FAILURE;
  }
  std::optional<std::string> cubin_dir_opt;
  if (cubin_dir != nullptr) {
    cubin_dir_opt.emplace(cubin_dir);
  }
  // Rebuild the std map on the DSO side of the boundary so no std container
  // crosses the ABI; the entries are C-compatible (name + AtenTensorHandle).
  std::unordered_map<std::string, AtenTensorHandle> constants;
  constants.reserve(num_constant_entries);
  for (size_t i = 0; i < num_constant_entries; ++i) {
    constants.emplace(constant_entries[i].name, constant_entries[i].handle);
  }
  auto* container = new torch::aot_inductor::AOTInductorModelContainer(
      num_models,
      std::string(device_str),
      constants,
      cubin_dir_opt);
  *container_handle =
      reinterpret_cast<AOTInductorModelContainerHandle>(container);
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerDelete(
    AOTInductorModelContainerHandle container_handle) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  delete container;
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerRun(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles, // array of input AtenTensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  AOTI_VECTOR_SIZE_CHECK(num_inputs, container->num_inputs(), "inputs");
  AOTI_VECTOR_SIZE_CHECK(num_outputs, container->num_outputs(), "outputs");

  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  AOTINoGradGuard guard;
  container->run(
      input_handles, output_handles, stream, proxy_executor_handle);
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerRunSingleThreaded(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles, // array of input AtenTensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  AOTI_VECTOR_SIZE_CHECK(num_inputs, container->num_inputs(), "inputs");
  AOTI_VECTOR_SIZE_CHECK(num_outputs, container->num_outputs(), "outputs");

  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  AOTINoGradGuard guard;
  container->run_single_threaded(
      input_handles, output_handles, stream, proxy_executor_handle);
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerGetNumConstants(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  *num_constants = container->num_constants();
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerGetConstantName(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** name) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  *name = container->constant_name(idx);
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerGetConstantOriginalFQN(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** original_fqn) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  *original_fqn = container->constant_original_fqn(idx);
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerGetConstantFromFolded(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    bool* from_folded) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  *from_folded = container->constant_from_folded(idx);
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerGetConstantType(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* type) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  *type = container->constant_type(idx);
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerGetConstantDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* dtype) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  *dtype = container->constant_dtype(idx);
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerGetConstantDataSize(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    size_t* data_size) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  *data_size = container->constant_data_size(idx);
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerExtractConstantsMap(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto constants_map =
      reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(
          constant_map_handle);
  const auto ret = container->extract_constants_map(use_inactive);
  for (const auto& pair : ret) {
    constants_map->emplace(pair.first, pair.second);
  }
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerExtractConstantsMapEntries(
    AOTInductorModelContainerHandle container_handle,
    const AOTInductorConstantMapEntry** entries,
    size_t* num_entries,
    bool use_inactive) AOTI_RUNTIME_TRY({
  if (!entries || !num_entries) {
    return AOTI_RUNTIME_FAILURE;
  }
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  const auto& extracted =
      container->extract_constants_map_entries(use_inactive);
  *entries = extracted.empty() ? nullptr : extracted.data();
  *num_entries = extracted.size();
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerUpdateUserManagedConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map =
      reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(
          constant_map_handle);
  container->update_constant_buffer(
      *input_map,
      use_inactive,
      validate_full_update,
      /* user_managed = */ true);
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerUpdateUserManagedConstantBufferPairs(
    AOTInductorModelContainerHandle container_handle,
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs,
    bool use_inactive,
    bool validate_full_update) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  // Build a local unordered_map inside
  std::unordered_map<std::string, AtenTensorHandle> input_map;
  input_map.reserve(num_pairs);
  for (size_t i = 0; i < num_pairs; ++i) {
    input_map.emplace(pairs[i].name, pairs[i].handle);
  }
  container->update_constant_buffer(
      input_map, use_inactive, validate_full_update, /*user_managed=*/true);
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map =
      reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(
          constant_map_handle);
  container->update_constant_buffer(
      *input_map, use_inactive, validate_full_update);
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBufferPairs(
    AOTInductorModelContainerHandle container_handle,
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs,
    bool use_inactive,
    bool validate_full_update) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = constant_map_from_pairs(pairs, num_pairs);
  container->update_constant_buffer(
      input_map, use_inactive, validate_full_update);
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBufferFromCpu(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map =
      reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(
          constant_map_handle);
  container->update_constant_buffer(
      *input_map,
      use_inactive,
      validate_full_update,
      /*user_managed=*/false,
      /*allow_h2d_copy=*/true);
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBufferFromCpuPairs(
    AOTInductorModelContainerHandle container_handle,
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs,
    bool use_inactive,
    bool validate_full_update) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = constant_map_from_pairs(pairs, num_pairs);
  container->update_constant_buffer(
      input_map,
      use_inactive,
      validate_full_update,
      /*user_managed=*/false,
      /*allow_h2d_copy=*/true);
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerUpdateInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle) AOTI_RUNTIME_TRY({
  return AOTInductorModelContainerUpdateConstantBuffer(
      container_handle,
      constant_map_handle,
      /*use_inactive=*/true,
      /*validate_full_update=*/true);
})

AOTIRuntimeError AOTInductorModelContainerUpdateInactiveConstantBufferPairs(
    AOTInductorModelContainerHandle container_handle,
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs) AOTI_RUNTIME_TRY({
  return AOTInductorModelContainerUpdateConstantBufferPairs(
      container_handle,
      pairs,
      num_pairs,
      /*use_inactive=*/true,
      /*validate_full_update=*/true);
})

AOTIRuntimeError AOTInductorModelContainerFreeInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  container->free_inactive_constant_buffer();
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerRunConstantFolding(
    AOTInductorModelContainerHandle container_handle,
    bool use_inactive,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  AOTINoGradGuard guard;
  container->run_const_fold(use_inactive, stream, proxy_executor_handle);
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerSwapConstantBuffer(
    AOTInductorModelContainerHandle container_handle) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  container->swap_constant_buffer();
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerGetNumInputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_inputs) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  *ret_num_inputs = container->num_inputs();
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerGetInputName(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** ret_input_names) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  *ret_input_names = container->input_name(input_idx);
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerGetNumOutputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_outputs) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  *ret_num_outputs = container->num_outputs();
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerGetOutputName(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const char** ret_output_names) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  *ret_output_names = container->output_name(output_idx);
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerGetCallSpec(
    AOTInductorModelContainerHandle container_handle,
    const char** in_spec,
    const char** out_spec) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  *in_spec = container->get_in_spec();
  *out_spec = container->get_out_spec();
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelCreate(
    AOTInductorModelHandle* model_handle,
    AOTInductorConstantMapHandle constant_map_handle) AOTI_RUNTIME_TRY({
  return createModelImpl(
      model_handle, constant_map_handle == nullptr, [=](auto& constant_map) {
        auto* input_map = reinterpret_cast<
            std::unordered_map<std::string, AtenTensorHandle>*>(
            constant_map_handle);
        if (input_map) {
          for (const auto& kv : *input_map) {
            constant_map.emplace(kv.first, kv.second);
          }
        }
      });
})

AOTIRuntimeError AOTInductorModelCreateV2(
    AOTInductorModelHandle* model_handle,
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs) AOTI_RUNTIME_TRY({
  return createModelImpl(
      model_handle, pairs == nullptr || num_pairs == 0, [=](auto& constant_map) {
        if (pairs && num_pairs > 0) {
          constant_map.reserve(num_pairs);
          for (size_t i = 0; i < num_pairs; ++i) {
            constant_map.emplace(pairs[i].name, pairs[i].handle);
          }
        }
      });
})

AOTIRuntimeError AOTInductorModelRun(
    AOTInductorModelHandle model_handle,
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles) AOTI_RUNTIME_TRY({
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  AOTINoGradGuard guard;
  model->run_impl(
      input_handles,
      output_handles,
      (torch::aot_inductor::DeviceStreamType) nullptr,
      nullptr);
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelDelete(
    AOTInductorModelHandle model_handle) AOTI_RUNTIME_TRY({
  auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(
      model_handle);
  delete model;
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelGetNumOutputs(
    AOTInductorModelHandle model_handle,
    size_t* ret_num_outputs) AOTI_RUNTIME_TRY({
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  *ret_num_outputs = model->num_outputs();
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelUpdateConstantsMap(
    AOTInductorModelHandle model_handle,
    AOTInductorConstantMapHandle constant_map_handle) AOTI_RUNTIME_TRY({
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
  auto input_map =
      reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(
          constant_map_handle);

  for (auto const& kv : *input_map) {
    constant_map->emplace(kv.first, kv.second);
  }
  model->update_constants_map(std::move(constant_map));
  return AOTI_RUNTIME_SUCCESS;
})

// C-ABI-safe variant: uses an array of (name, handle) pairs instead of an
// opaque pointer to std::unordered_map, so the host and DSO can use
// different C++ standard libraries without ABI conflicts.
AOTIRuntimeError AOTInductorModelUpdateConstantsMapV2(
    AOTInductorModelHandle model_handle,
    const AOTInductorConstantMapEntry* pairs,
    int32_t num_pairs) AOTI_RUNTIME_TRY({
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
  constant_map->reserve(num_pairs);
  for (int32_t i = 0; i < num_pairs; ++i) {
    constant_map->emplace(pairs[i].name, pairs[i].handle);
  }
  model->update_constants_map(std::move(constant_map));
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerGetConstantsBlobSize(
    AOTInductorModelContainerHandle container_handle,
    uint64_t* ret_size) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  *ret_size = container->constant_blob_size();
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorModelContainerDidCallLoadConstants(
    AOTInductorModelContainerHandle container_handle,
    bool* did_call_load_constants) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  *did_call_load_constants = container->did_call_load_constants();
  return AOTI_RUNTIME_SUCCESS;
})


// Load weights from a single blob in weight_blob_ptr
AOTIRuntimeError AOTInductorModelUpdateConstantsFromBlob(
    AOTInductorModelContainerHandle container_handle,
    const uint8_t* weight_blob_ptr) AOTI_RUNTIME_TRY({
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  container->update_constants_from_blob(weight_blob_ptr);
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorSetUsePinnedAsyncConstantsCopy(
    bool enabled) AOTI_RUNTIME_TRY({
  torch::aot_inductor::setUsePinnedAsyncConstantsCopy(enabled);
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorSetPinnedAsyncConstantsCopyStageBufferBytes(
    size_t bytes) AOTI_RUNTIME_TRY({
  torch::aot_inductor::setPinnedAsyncConstantsCopyStageBufferBytes(bytes);
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorSetPinnedAsyncConstantsCopyCpuThreads(
    size_t threads) AOTI_RUNTIME_TRY({
  torch::aot_inductor::setPinnedAsyncConstantsCopyCpuThreads(threads);
  return AOTI_RUNTIME_SUCCESS;
})

AOTIRuntimeError AOTInductorGetLastError(
    const char** error_msg) {
  *error_msg = g_aoti_last_error.c_str();
  return AOTI_RUNTIME_SUCCESS;
}

} // extern "C"

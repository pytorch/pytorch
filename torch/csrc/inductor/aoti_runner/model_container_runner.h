#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/inductor/aoti_runner/model_container_observer.h>
#include <torch/csrc/inductor/aoti_runtime/interface.h>
#include <torch/csrc/inductor/aoti_torch/proxy_executor.h>

#include <atomic>
#include <memory>

// Forward declare DynamicLibrary
namespace at {
struct DynamicLibrary;
}

namespace torch::inductor {
using TensorConstantMap = std::unordered_map<std::string, at::Tensor*>;

class TORCH_API AOTIModelContainerRunner {
 public:
  AOTIModelContainerRunner(const AOTIModelContainerRunner& other) = delete;
  AOTIModelContainerRunner(AOTIModelContainerRunner&& other) = delete;
  AOTIModelContainerRunner& operator=(const AOTIModelContainerRunner& other) =
      delete;
  AOTIModelContainerRunner& operator=(AOTIModelContainerRunner&& other) =
      delete;
  virtual ~AOTIModelContainerRunner();

  std::vector<at::Tensor> run(
      const std::vector<at::Tensor>& inputs,
      void* stream_handle = nullptr);

  // boxed_run will steal the ownership of the input tensors
  std::vector<at::Tensor> boxed_run(
      std::vector<at::Tensor>&& inputs,
      void* stream_handle = nullptr);

  std::unordered_map<std::string, std::string> getConstantNamesToOriginalFQNs()
      const;
  std::unordered_map<std::string, int32_t> getConstantNamesToDtypes() const;

  const std::unordered_map<std::string, at::Tensor> extract_constants_map(
      bool use_inactive) const;
  void update_inactive_constant_buffer(const TensorConstantMap& const_map);
  void update_constant_buffer(
      std::unordered_map<std::string, at::Tensor>& tensor_map,
      bool use_inactive,
      bool validate_full_updates,
      bool user_managed = false);
  void update_constant_buffer(
      const TensorConstantMap& const_map,
      bool use_inactive,
      bool validate_full_updates,
      bool user_managed = false);
  // Update constants from CPU tensors. CPU tensors are silently copied to the
  // model's device.
  void update_constant_buffer_from_cpu(
      std::unordered_map<std::string, at::Tensor>& tensor_map,
      bool use_inactive,
      bool validate_full_updates);
  void update_constant_buffer_from_cpu(
      const TensorConstantMap& const_map,
      bool use_inactive,
      bool validate_full_updates);
  void run_const_fold(
      bool use_inactive,
      AOTInductorStreamHandle cuda_stream_handle = nullptr);
  void swap_constant_buffer();
  void free_inactive_constant_buffer();
  void update_constant_buffer_from_blob(const std::string& weights_path);
  bool did_call_load_constants() const;

  std::vector<std::string> get_call_spec();
  std::vector<std::string> get_input_names();
  std::vector<std::string> get_output_names();

  // Returns the torchbind custom-class constants embedded in the loaded
  // model. The IValue payloads alias the live entries inside the proxy
  // executor: downcasting to a CustomClassHolder subclass and mutating
  // its state will affect subsequent run() invocations. Returns empty when
  // the model has no torchbind constants.
  std::unordered_map<std::string, c10::IValue> get_custom_objs() const {
    return proxy_executor_ ? proxy_executor_->get_custom_objs()
                           : std::unordered_map<std::string, c10::IValue>{};
  }

  // Attach an observer to receive begin/end callbacks bracketing container
  // lifecycle events (constants load, constant-buffer update/swap/fold/free,
  // inference). Optional; a null observer is zero overhead.
  //
  // Attach-once, and detaching is NOT supported: the hot path reads the
  // observer through an atomic raw pointer, and because the owning shared_ptr
  // is never replaced that pointer stays valid for the lifetime of the runner.
  // Allowing a swap or a detach would race with -- and could free the observer
  // out from under -- an in-flight run(). The slot is claimed with a
  // compare-exchange rather than a plain check so a second concurrent caller
  // reliably throws instead of racing on observer_owner_; only the winner
  // assigns it. Passing nullptr is always an unconditional no-op.
  void set_observer(std::shared_ptr<AOTIModelContainerObserver> observer) {
    if (observer == nullptr) {
      return;
    }
    AOTIModelContainerObserver* expected = nullptr;
    TORCH_CHECK(
        observer_.compare_exchange_strong(
            expected,
            observer.get(),
            std::memory_order_release,
            std::memory_order_relaxed),
        "AOTIModelContainerRunner::set_observer() may only be called once per runner");
    // Safe to publish before taking ownership: the caller's shared_ptr keeps
    // the observer alive across this call.
    observer_owner_ = std::move(observer);
  }

 protected:
  AOTIModelContainerRunner(
      const std::string& model_so_path,
      size_t num_models,
      const std::string& device_str,
      const std::string& cubin_dir,
      const bool run_single_threaded);

  // Construct with externally-provided weights. Skips the .so weight load
  // entirely (no GPU allocation) and seeds the container's constants map
  // from the caller-supplied tensors. The caller retains ownership.
  AOTIModelContainerRunner(
      const std::string& model_so_path,
      size_t num_models,
      const std::string& device_str,
      const std::string& cubin_dir,
      std::unordered_map<std::string, at::Tensor>& constants);

  // Default constructor for custom device implementations that don't
  // use .so files. Derived classes must override run_impl().
  AOTIModelContainerRunner();

  virtual std::vector<at::Tensor> run_impl(
      std::vector<AtenTensorHandle>& input_handles,
      void* stream_handle);

  std::unique_ptr<at::DynamicLibrary> model_so_;
  decltype(&AOTInductorModelContainerCreateWithDevice) create_func_{nullptr};
  decltype(&AOTInductorModelContainerDelete) delete_func_{nullptr};
  decltype(&AOTInductorModelContainerGetNumInputs) get_num_inputs_func_{
      nullptr};
  decltype(&AOTInductorModelContainerGetInputName) get_input_name_func_{
      nullptr};
  decltype(&AOTInductorModelContainerGetNumOutputs) get_num_outputs_func_{
      nullptr};
  decltype(&AOTInductorModelContainerGetOutputName) get_output_name_func_{
      nullptr};
  decltype(&AOTInductorModelContainerRun) run_func_{nullptr};
  decltype(&AOTInductorModelContainerGetNumConstants) get_num_constants_func_{
      nullptr};
  decltype(&AOTInductorModelContainerGetConstantName) get_constant_name_func_{
      nullptr};
  decltype(&AOTInductorModelContainerGetConstantOriginalFQN)
      get_constant_original_fqn_func_{nullptr};
  decltype(&AOTInductorModelContainerGetConstantDtype) get_constant_dtype_func_{
      nullptr};
  decltype(&AOTInductorModelContainerExtractConstantsMap)
      extract_constants_map_func_{nullptr};
  decltype(&AOTInductorModelContainerUpdateUserManagedConstantBuffer)
      update_user_managed_constant_buffer_func_{nullptr};
  decltype(&AOTInductorModelContainerUpdateConstantBuffer)
      update_constant_buffer_func_{nullptr};
  decltype(&AOTInductorModelContainerUpdateConstantBufferFromCpu)
      update_constant_buffer_from_cpu_func_{nullptr};
  decltype(&AOTInductorModelContainerUpdateInactiveConstantBuffer)
      update_inactive_constant_buffer_func_{nullptr};
  decltype(&AOTInductorModelContainerRunConstantFolding) run_const_fold_func_{
      nullptr};
  decltype(&AOTInductorModelContainerSwapConstantBuffer)
      swap_constant_buffer_func_{nullptr};
  decltype(&AOTInductorModelContainerFreeInactiveConstantBuffer)
      free_inactive_constant_buffer_func_{nullptr};
  decltype(&AOTInductorModelContainerGetCallSpec) get_call_spec_func_{nullptr};
  decltype(&AOTInductorModelContainerGetConstantsBlobSize)
      get_constants_blob_size_func_{nullptr};
  decltype(&AOTInductorModelContainerDidCallLoadConstants)
      did_call_load_constants_func_{nullptr};
  decltype(&AOTInductorModelUpdateConstantsFromBlob)
      update_constants_from_blob_func_{nullptr};
  decltype(&AOTInductorGetLastError) get_last_error_func_{nullptr};
  decltype(&AOTInductorModelContainerCreateWithExternalConstants)
      create_with_external_constants_func_{nullptr};

  AOTInductorModelContainerHandle container_handle_ = nullptr;

  AOTIProxyExecutorHandle proxy_executor_handle_ = nullptr;

  // Read on the hot path; see set_observer() for why this is attach-once.
  AOTIModelContainerObserver* observer() const {
    return observer_.load(std::memory_order_acquire);
  }

 private:
  void load_aoti_symbols(
      const std::string& model_so_path,
      const std::string& device_str,
      bool run_single_threaded);

  std::unique_ptr<torch::aot_inductor::ProxyExecutor> proxy_executor_;

  // Private, not protected: the lifetime argument in set_observer() only holds
  // while every write goes through that compare-exchange, so subclasses (and
  // out-of-tree runners registered via RegisterAOTIModelRunner) must not be
  // able to assign these directly. They get the observer() accessor instead.
  std::shared_ptr<AOTIModelContainerObserver> observer_owner_;
  std::atomic<AOTIModelContainerObserver*> observer_{nullptr};
};

using CreateAOTIModelRunnerFunc = std::unique_ptr<AOTIModelContainerRunner> (*)(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& bin_dir,
    const bool run_single_threaded);

// Return a global map "device name" -> "aoti model runner create function" for
// all registered in AOTI external backends
TORCH_API std::unordered_map<std::string, CreateAOTIModelRunnerFunc>&
getAOTIModelRunnerRegistry();

// To register a new external backend in AOTI one needs to create an instance of
// this struct. It is not thread-safe. Because it is expected to be called
// during the initialization of the program.
struct TORCH_API RegisterAOTIModelRunner{RegisterAOTIModelRunner(
    const std::string& name,
    CreateAOTIModelRunnerFunc create_aoti_model_runner_fn){
    getAOTIModelRunnerRegistry()[name] = create_aoti_model_runner_fn;
} // namespace torch::inductor
}
;

} // namespace torch::inductor
#endif

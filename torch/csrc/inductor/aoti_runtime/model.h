#pragma once

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#include <torch/csrc/inductor/aoti_runtime/model_base.h>
#ifdef USE_CUDA
// Header-only over the stable C ABI (shim.h), so it is safe to include here per
// the rule above. Provides the per-instance cuda-graph tree manager member below.
#include <torch/csrc/inductor/aoti_runtime/cudagraph_tree.h>
#endif

struct AOTInductorArrayRefTensor;

namespace torch::aot_inductor {

class AOTInductorModel : public AOTInductorModelBase<AOTInductorModel> {
 public:
  AOTInductorModel(
      std::shared_ptr<ConstantMap> constants_map,
      std::shared_ptr<std::vector<ConstantHandle>> constants_array,
      const std::string& device_str,
      std::optional<std::string> cubin_dir);

  std::unordered_map<std::string, AtenTensorHandle> const_run_impl(
      DeviceStreamType stream,
      AOTIProxyExecutorHandle proxy_executor,
      bool initialization = false);

  void _const_run_impl(
      std::vector<AtenTensorHandle>& output_handles,
      DeviceStreamType stream,
      AOTIProxyExecutorHandle proxy_executor);

  void run_impl(
      AtenTensorHandle*
          input_handles, // array of input AtenTensorHandle; handles
                         // are stolen; the array itself is borrowed
      AtenTensorHandle*
          output_handles, // array for writing output AtenTensorHandle; handles
                          // will be stolen by the caller; the array itself is
                          // borrowed
      DeviceStreamType stream,
      AOTIProxyExecutorHandle proxy_executor);

  template <typename Inputs, typename Outputs>
  Outputs run_impl_minimal_arrayref_interface(
      const Inputs& inputs,
      DeviceStreamType stream,
      AOTIProxyExecutorHandle proxy_executor);

  void run_impl_minimal_arrayref_interface_v2_raw(
      const AOTInductorArrayRefTensor* c_inputs,
      AOTInductorArrayRefTensor* c_outputs,
      DeviceStreamType stream,
      AOTIProxyExecutorHandle proxy_executor);

  static std::unique_ptr<AOTInductorModel> Create(
      std::shared_ptr<ConstantMap> constants_map,
      std::shared_ptr<std::vector<ConstantHandle>> constants_array,
      const std::string& device_str,
      std::optional<std::string> cubin_dir) {
    return std::make_unique<AOTInductorModel>(
        std::move(constants_map),
        std::move(constants_array),
        device_str,
        std::move(cubin_dir));
  }

 private:
  std::unique_ptr<AOTInductorModelKernelsBase> kernels_;
#ifdef USE_CUDA
  // Per-instance cuda-graph manager: owns THIS model's private graph pool
  // + capture stream, so concurrent instances in a model_container are isolated.
  // Lazily created by the generated run_impl on the first captured partition;
  // stays null for models with no cuda-graph partitions. Destroyed with the
  // model (no leak). See cudagraph_tree.h.
  std::unique_ptr<AOTICUDAGraphTreeManager> cudagraph_mgr_;
  // Per-instance cuda-graph memory_planning slab cache. Keyed by a packed
  // (pool id, partition id) int64; each value owns the persistent slab whose
  // base address the captured partitions bake into their reinterpret_tensor
  // views, so the address stays stable across forwards. The slab is sized to the
  // dynamic-shape upper bounds and shared across shapes (single max slab). Lives
  // per AOTInductorModel instance (NOT process-static) so concurrent instances
  // in a model_container have isolated slabs; the RAII handles free the slabs at
  // model destruction. Populated by the generated memory_planning codegen (see
  // _codegen_create_cudagraph_cached in memory_planning.py).
  std::unordered_map<int64_t, RAIIAtenTensorHandle> cudagraph_slabs_;
#endif
};

} // namespace torch::aot_inductor

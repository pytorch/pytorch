#pragma once

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#include <torch/csrc/inductor/aoti_runtime/model_base.h>

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
};

} // namespace torch::aot_inductor

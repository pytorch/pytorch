#pragma once

#include <torch/csrc/inductor/aoti_runtime/model.h>


namespace torch::aot_inductor {

class AOTInductorModelClassNamePlaceholder : public AOTInductorModelBase<AOTInductorModelClassNamePlaceholder> {
  public:
   AOTInductorModelClassNamePlaceholder(
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
 
   static std::unique_ptr<AOTInductorModelClassNamePlaceholder> Create(
       std::shared_ptr<ConstantMap> constants_map,
       std::shared_ptr<std::vector<ConstantHandle>> constants_array,
       const std::string& device_str,
       std::optional<std::string> cubin_dir) {
     return std::make_unique<AOTInductorModelClassNamePlaceholder>(
         std::move(constants_map),
         std::move(constants_array),
         device_str,
         std::move(cubin_dir));
   }
 
  private:
   std::unique_ptr<AOTInductorModelKernelsBase> kernels_;
 };

 } // namespace torch::aot_inductor

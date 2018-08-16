// #if !(defined _WIN32)
// #pragma once

// #include "torch/csrc/jit/fusers/fuser_interface.h"

// #include "torch/csrc/jit/fusers/cpu/concat_desc.h"
// #include "torch/csrc/jit/fusers/cpu/temp_file.h"
// #include "torch/csrc/jit/fusers/cpu/fusion_compiler_config.h"
// #include "torch/csrc/jit/fusers/cpu/annotated_graph.h"

// #include "torch/csrc/jit/ir.h"
// #include "torch/csrc/jit/assertions.h"

// #include "torch/csrc/WindowsTorchApiMacro.h"
// #include "torch/csrc/utils/disallow_copy.h"

// #include "ATen/ATen.h"

// #include <string>
// #include <vector>
// #include <sstream>
// #include <memory>

// namespace torch { namespace jit { namespace cpufuser {

// TORCH_API struct CPUFusionFunction : public torch::jit::FusionFunction {
//   TH_DISALLOW_COPY_AND_ASSIGN(CPUFusionFunction);

//   CPUFusionFunction(
//     const std::string& name
//   , AnnotatedGraph& agraph
//   , CPUFusionCompilerConfig& config);

//   // Note: Creates new tensors for outputs
//   void launch(
//     at::ArrayRef<at::Tensor> inputs
//   , std::vector<at::Tensor>& outputs);

//   // Note: expects outputs to be pre-allocated
//   void launch_with_tensors(
//     at::ArrayRef<at::Tensor> inputs
//   , at::ArrayRef<at::Tensor> outputs);

//   const std::vector<TensorDesc>& outputDescriptors() const {
//     return output_desc;
//   }

// private:
//   at::Backend backend() const { 
//     return at::kCPU; 
//   }

//   uint64_t get_rand_offset(uint32_t numel) {
//     return numel;
//   }

//   // arguments is a list of pointers to the arguments for the compiled CUDA/CPU
//   // code.
//   // The format of arguments is suitable for directly passing to a call to
//   // cuLaunchKernel as the kernel arguments.
//   // Currently the first argument is a pointer to numel (for passing to
//   // CUDA code), and the remainder are pointers to the TensorInfo<T> structs
//   // that compiled code uses to load Tensor data.
//   // launch_with_tensors handles packing at::Tensors into this arguments array.
//   // CPU code uses the same convension so that launch_with_tensors can be shared.
//   void launch_raw(uint32_t numel, void** arguments) {
//     kernel(numel, arguments);
//   }

//   std::unique_ptr<DynamicLibrary> so_lib;
//   void (*kernel)(uint32_t, void**) = nullptr;

//   bool has_random;
//   std::string name;
//   // We keep these around for debugging
//   std::string compilation_unit;
//   std::vector<TensorDesc> input_desc;
//   std::vector<TensorDesc> output_desc;

//   // same size as output_desc, describes whether
//   // an output is actually a concatenation of
//   // many subtensors that the fusion group produces
//   std::vector<ConcatDesc> concat_desc;
// };

// } // namespace cpufuser
// } // namespace jit
// } // namespace torch

// #endif // !(defined _WIN32)
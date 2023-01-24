#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>
#include <torch/extension.h>

#include <memory>

using namespace torch::jit::fuser::cuda;

at::Tensor sinh_nvfuser(const at::Tensor& input) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int dim = input.dim();
  auto dtype = input.scalar_type();
  auto x =
      TensorViewBuilder().ndims(dim).dtype(aten_to_data_type(dtype)).build();
  fusion.addInput(x);

  // Using equation sinh(x) = [ exp(x) - exp(-1) ] / 2
  auto output = div(sub(exp(x), exp(neg(x))), IrBuilder::create<Double>(2.0));
  fusion.addOutput(output);

  std::cout << "Create fusion:" << std::endl;
  fusion.print();

  auto lparams = schedulePointwise(&fusion, {input});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input}, lparams);
  auto outputs = fe.runFusion({input}, lparams);

  return outputs[0];
}

TORCH_LIBRARY(myop, m) {
  m.def("sinh_nvfuser", sinh_nvfuser);
}

TORCH_LIBRARY_IMPL(myop, CUDA, m) {
  m.impl("sinh_nvfuser", sinh_nvfuser);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

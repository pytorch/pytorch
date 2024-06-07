#include <test/cpp/lazy/test_lazy_ops_util.h>

#include <torch/csrc/lazy/backend/lowering_context.h>
#include <torch/csrc/lazy/core/ir_builder.h>
#include <torch/csrc/lazy/core/ir_dump_util.h>
#include <torch/csrc/lazy/core/tensor_impl.h>

#include <iostream>
#include <string>

namespace torch {
namespace lazy {
namespace {

std::unordered_set<std::string>* CreateIgnoredCounters() {
  std::unordered_set<std::string>* icounters =
      new std::unordered_set<std::string>();
  // Add below the counters whose name need to be ignored when doing
  // is-any-counter-changed assertions.
  icounters->insert("aten::rand");
  return icounters;
}

} // namespace

const std::unordered_set<std::string>* GetIgnoredCounters() {
  static const std::unordered_set<std::string>* icounters =
      CreateIgnoredCounters();
  return icounters;
}

at::Tensor ToCpuTensor(const at::Tensor& tensor) {
  // tensor.to() implicitly triggers a sync if t.device=torch::kLazy.
  return tensor.to(torch::kCPU);
}

torch::Tensor CopyToDevice(
    const torch::Tensor& tensor,
    const torch::Device& device) {
  return tensor.clone().to(device, /*non_blocking=*/false, /*copy=*/true);
}

bool EqualValues(at::Tensor tensor1, at::Tensor tensor2) {
  tensor1 = ToCpuTensor(tensor1);
  tensor2 = ToCpuTensor(tensor2);
  if (torch::isnan(tensor1).any().item<bool>()) {
    EXPECT_TRUE(EqualValues(torch::isnan(tensor1), torch::isnan(tensor2)));
    tensor1.nan_to_num_();
    tensor2.nan_to_num_();
  }
  if (tensor1.sizes() != tensor2.sizes() ||
      tensor1.dtype() != tensor2.dtype()) {
    std::cerr << "Different shape:\n"
              << tensor1.dtype() << " " << tensor1.sizes() << "\n-vs-\n"
              << tensor2.dtype() << " " << tensor2.sizes() << "\n";
    return false;
  }
  at::ScalarType type1 = tensor1.scalar_type();
  at::ScalarType type2 = tensor2.scalar_type();
  if (type1 != type2) {
    tensor1 = tensor1.toType(type2);
  }
  bool equal = tensor1.equal(tensor2);
  return equal;
}

bool EqualValuesNoElementTypeCheck(at::Tensor tensor1, at::Tensor tensor2) {
  tensor1 = ToCpuTensor(tensor1);
  tensor2 = ToCpuTensor(tensor2);
  if (tensor1.sizes() != tensor2.sizes()) {
    std::cerr << "Different shape:\n"
              << tensor1.dtype() << " " << tensor1.sizes() << "\n-vs-\n"
              << tensor2.dtype() << " " << tensor2.sizes() << "\n";
    return false;
  }
  at::ScalarType type1 = tensor1.scalar_type();
  at::ScalarType type2 = tensor2.scalar_type();
  if (type1 != type2) {
    tensor1 = tensor1.toType(type2);
  }
  bool equal = tensor1.equal(tensor2);
  return equal;
}

void ForEachDevice(const std::function<void(const torch::Device&)>& devfn) {
  // Currently TorchScript backend only supports one type of hardware per
  // process, which is set by env. And the ordinal is always 0 given distributed
  // training/ multi-device is not supported yet.
  auto device = torch::lazy::BackendDevice();
  torch::Device torch_device = torch::lazy::backendDeviceToAtenDevice(device);
  devfn(torch_device);
}

bool CloseValues(
    at::Tensor tensor1,
    at::Tensor tensor2,
    double rtol,
    double atol) {
  tensor1 = ToCpuTensor(tensor1);
  tensor2 = ToCpuTensor(tensor2);
  if (torch::isnan(tensor1).any().item<bool>()) {
    EXPECT_TRUE(EqualValues(torch::isnan(tensor1), torch::isnan(tensor2)));
    tensor1.nan_to_num_();
    tensor2.nan_to_num_();
  }
  if (tensor1.sizes() != tensor2.sizes() ||
      tensor1.dtype() != tensor2.dtype()) {
    std::cerr << "Different shape:\n"
              << tensor1.dtype() << " " << tensor1.sizes() << "\n-vs-\n"
              << tensor2.dtype() << " " << tensor2.sizes() << "\n";
    return false;
  }
  bool equal = tensor1.allclose(tensor2, rtol, atol);
  return equal;
}

std::string GetTensorTextGraph(at::Tensor tensor) {
  torch::lazy::LazyTensorPtr lazy_tensor = torch::lazy::TryGetLtcTensor(tensor);
  return torch::lazy::DumpUtil::ToText({lazy_tensor->GetIrValue().node.get()});
}

std::string GetTensorDotGraph(at::Tensor tensor) {
  torch::lazy::LazyTensorPtr lazy_tensor = torch::lazy::TryGetLtcTensor(tensor);
  return torch::lazy::DumpUtil::ToDot({lazy_tensor->GetIrValue().node.get()});
}

void TestBackward(
    const std::vector<torch::Tensor>& inputs,
    const torch::Device& device,
    const std::function<torch::Tensor(const std::vector<torch::Tensor>&)>&
        testfn,
    double rtol,
    double atol,
    int derivative_level) {
  std::vector<torch::Tensor> input_vars;
  std::vector<torch::Tensor> xinput_vars;
  std::vector<torch::Tensor> inputs_w_grad;
  std::vector<torch::Tensor> xinputs_w_grad;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const torch::Tensor& input = inputs[i];
    if (input.defined()) {
      torch::Tensor oinput =
          input.clone().detach().set_requires_grad(input.requires_grad());
      input_vars.push_back(oinput);

      torch::Tensor xinput = CopyToDevice(input, device)
                                 .detach()
                                 .set_requires_grad(input.requires_grad());
      xinput_vars.push_back(xinput);
      if (input.requires_grad()) {
        inputs_w_grad.push_back(oinput);
        xinputs_w_grad.push_back(xinput);
      }
    } else {
      input_vars.emplace_back();
      xinput_vars.emplace_back();
    }
  }

  torch::Tensor output = testfn(input_vars);
  torch::Tensor xoutput = testfn(xinput_vars);
  torch::lazy::AllClose(output, xoutput, rtol, atol);

  std::vector<torch::Tensor> outs = {output};
  std::vector<torch::Tensor> xouts = {xoutput};
  for (int d = 1; d <= derivative_level; ++d) {
    // Check grad of sum(outs) w.r.t inputs_w_grad.
    torch::Tensor sum = torch::zeros_like(outs[0]).sum();
    torch::Tensor xsum = torch::zeros_like(xouts[0]).sum();
    for (size_t i = 0; i < outs.size(); ++i) {
      if (outs[i].requires_grad()) {
        sum += outs[i].sum();
        xsum += xouts[i].sum();
      }
    }
    // Calculating higher order derivative requires create_graph=true
    bool create_graph = d != derivative_level;
    outs = torch::autograd::grad(
        {sum},
        inputs_w_grad,
        /*grad_outputs=*/{},
        /*retain_graph=*/c10::nullopt,
        /*create_graph=*/create_graph,
        /*allow_unused=*/true);
    xouts = torch::autograd::grad(
        {xsum},
        xinputs_w_grad,
        /*grad_outputs=*/{},
        /*retain_graph=*/c10::nullopt,
        /*create_graph=*/create_graph,
        /*allow_unused=*/true);
    for (size_t i = 0; i < outs.size(); ++i) {
      ASSERT_EQ(outs[i].defined(), xouts[i].defined());
      if (outs[i].defined()) {
        AllClose(outs[i], xouts[i], rtol, atol);
      }
    }
  }
}

} // namespace lazy
} // namespace torch

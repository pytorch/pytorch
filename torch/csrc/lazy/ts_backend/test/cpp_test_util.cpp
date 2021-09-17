#include "cpp_test_util.h"

#include <iostream>
#include <string>

#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"
#include "lazy_tensor_core/csrc/ir_dump_util.h"
#include "lazy_tensor_core/csrc/lowering_context.h"
#include "lazy_tensor_core/csrc/ops/device_data.h"
#include "lazy_tensor_core/csrc/tensor_impl.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/torch_util.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/sys_util.h"

namespace torch_lazy_tensors {
namespace cpp_test {
namespace {

void DumpDifferences(const at::Tensor& tensor1, const at::Tensor& tensor2) {
  static bool dump_tensors =
      lazy_tensors::sys_util::GetEnvBool("XLA_TEST_DUMP_TENSORS", false);
  at::Tensor dtensor1 = tensor1;
  at::Tensor dtensor2 = tensor2;
  if (tensor1.dtype() == at::kBool) {
    dtensor1 = tensor1.toType(at::kByte);
  }
  if (tensor2.dtype() == at::kBool) {
    dtensor2 = tensor2.toType(at::kByte);
  }
  at::Tensor diff = dtensor1 - dtensor2;
  std::cerr << "Difference Tensor:\n" << diff << "\n";
  if (dump_tensors) {
    std::cerr << "Compared Tensors:\n"
              << tensor1 << "\n-vs-\n"
              << tensor2 << "\n";
  }
}

void MaybeDumpGraph(const at::Tensor& tensor) {
  static std::string dump_graph =
      lazy_tensors::sys_util::GetEnvString("XLA_TEST_DUMP_GRAPHS", "");
  if (!dump_graph.empty() && bridge::IsLtcTensor(tensor)) {
    std::string graph_str;
    if (dump_graph == "text") {
      graph_str = GetTensorTextGraph(tensor);
    } else if (dump_graph == "dot") {
      graph_str = GetTensorDotGraph(tensor);
    }
    if (!graph_str.empty()) {
      std::cerr << "\n>> Tensor Graph:\n" << graph_str << "\n\n";
    }
  }
}

std::unordered_set<std::string>* CreateIgnoredCounters() {
  std::unordered_set<std::string>* icounters =
      new std::unordered_set<std::string>();
  // Add below the counters whose name need to be ignored when doing
  // is-any-counter-changed assertins.
  icounters->insert("aten::rand");
  return icounters;
}

}  // namespace

const std::unordered_set<std::string>* GetIgnoredCounters() {
  static const std::unordered_set<std::string>* icounters =
      CreateIgnoredCounters();
  return icounters;
}

at::Tensor ToCpuTensor(const at::Tensor& tensor) {
  // tensor.to() implicitly triggers a sync if t.device=torch::kLazy.
  return tensor.to(torch::kCPU);
}

torch::Tensor CopyToDevice(const torch::Tensor& tensor,
                           const torch::Device& device) {
  return tensor.clone().to(device, /*non_blocking=*/false, /*copy=*/true);
}

bool EqualValues(at::Tensor tensor1, at::Tensor tensor2) {
  MaybeDumpGraph(tensor1);
  MaybeDumpGraph(tensor2);
  tensor1 = ToCpuTensor(tensor1);
  tensor2 = ToCpuTensor(tensor2);
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
  if (!equal) {
    DumpDifferences(tensor1, tensor2);
  }
  return equal;
}

bool EqualValuesNoElementTypeCheck(at::Tensor tensor1, at::Tensor tensor2) {
  MaybeDumpGraph(tensor1);
  MaybeDumpGraph(tensor2);
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
  if (!equal) {
    DumpDifferences(tensor1, tensor2);
  }
  return equal;
}

void ForEachDevice(lazy_tensors::Span<const DeviceType> device_types,
                   const std::function<void(const Device&)>& devfn) {
  const Device* device = GetDefaultDevice();
  if (device_types.empty() ||
      std::find(device_types.begin(), device_types.end(), device->hw_type) !=
          device_types.end()) {
    bridge::SetCurrentDevice(*device);
    devfn(*device);
  } else {
    GTEST_SKIP();
  }
}

void ForEachDevice(lazy_tensors::Span<const DeviceType> device_types,
                   const std::function<void(const torch::Device&)>& devfn) {
  const Device* device = GetDefaultDevice();
  if (device_types.empty() ||
      std::find(device_types.begin(), device_types.end(), device->hw_type) !=
          device_types.end()) {
    torch::Device torch_device = bridge::LtcDeviceToAtenDevice(*device);
    bridge::SetCurrentDevice(torch_device);
    devfn(torch_device);
  } else {
    GTEST_SKIP();
  }
}

void ForEachDevice(const std::function<void(const Device&)>& devfn) {
  ForEachDevice({}, devfn);
}

void ForEachDevice(const std::function<void(const torch::Device&)>& devfn) {
  ForEachDevice({}, devfn);
}

bool CloseValues(at::Tensor tensor1, at::Tensor tensor2, double rtol,
                 double atol) {
  MaybeDumpGraph(tensor1);
  MaybeDumpGraph(tensor2);
  tensor1 = ToCpuTensor(tensor1);
  tensor2 = ToCpuTensor(tensor2);
  if (tensor1.sizes() != tensor2.sizes() ||
      tensor1.dtype() != tensor2.dtype()) {
    std::cerr << "Different shape:\n"
              << tensor1.dtype() << " " << tensor1.sizes() << "\n-vs-\n"
              << tensor2.dtype() << " " << tensor2.sizes() << "\n";
    return false;
  }
  bool equal = tensor1.allclose(tensor2, rtol, atol);
  if (!equal) {
    DumpDifferences(tensor1, tensor2);
  }
  return equal;
}

void WithAllDevices(
    lazy_tensors::Span<const DeviceType> device_types,
    const std::function<void(const std::vector<Device>&,
                             const std::vector<Device>&)>& devfn) {
  for (auto device_type : device_types) {
    std::vector<Device> devices;
    std::vector<Device> all_devices;
    for (const auto& device_str :
         lazy_tensors::ComputationClient::Get()->GetLocalDevices()) {
      Device device(device_str);
      if (device.hw_type == device_type) {
        devices.push_back(device);
      }
    }
    for (const auto& device_str :
         lazy_tensors::ComputationClient::Get()->GetAllDevices()) {
      Device device(device_str);
      if (device.hw_type == device_type) {
        all_devices.push_back(device);
      }
    }
    if (!devices.empty()) {
      devfn(devices, all_devices);
    }
  }
}

std::string GetTensorTextGraph(at::Tensor tensor) {
  LazyTensor xtensor = bridge::GetLtcTensor(tensor);
  return ir::DumpUtil::ToText({xtensor.GetIrValue().node.get()});
}

std::string GetTensorDotGraph(at::Tensor tensor) {
  LazyTensor xtensor = bridge::GetLtcTensor(tensor);
  return ir::DumpUtil::ToDot({xtensor.GetIrValue().node.get()});
}

void TestBackward(
    const std::vector<torch::Tensor>& inputs, const torch::Device& device,
    const std::function<torch::Tensor(const std::vector<torch::Tensor>&)>&
        testfn,
    double rtol, double atol, int derivative_level) {
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
  AllClose(output, xoutput, rtol, atol);

  std::vector<torch::Tensor> outs = {output};
  std::vector<torch::Tensor> xouts = {xoutput};
  for (int d = 1; d <= derivative_level; ++d) {
    // Check grad of sum(outs) w.r.t inputs_w_grad.
    torch::Tensor sum = torch::zeros_like(outs[0]).sum();
    torch::Tensor xsum = torch::zeros_like(xouts[0]).sum();
    for (int i = 0; i < outs.size(); ++i) {
      if (outs[i].requires_grad()) {
        sum += outs[i].sum();
        xsum += xouts[i].sum();
      }
    }
    // Calculating higher order derivative requires create_graph=true
    bool create_graph = d != derivative_level;
    outs = torch::autograd::grad({sum}, inputs_w_grad, /*grad_outputs=*/{},
                                 /*retain_graph=*/c10::nullopt,
                                 /*create_graph=*/create_graph,
                                 /*allow_unused=*/true);
    xouts = torch::autograd::grad({xsum}, xinputs_w_grad, /*grad_outputs=*/{},
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

}  // namespace cpp_test
}  // namespace torch_lazy_tensors

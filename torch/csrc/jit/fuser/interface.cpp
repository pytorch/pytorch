#include <ATen/ATen.h>
#include <aten/src/ATen/core/jit_type.h>
#include <ATen/core/stack.h>
#include <c10/core/DeviceType.h>
#include <c10/core/TensorOptions.h>

#include <torch/csrc/jit/ir.h>

// New fuser includes
#include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/fuser/cpu/interface.h>
#include <torch/csrc/jit/fuser/cuda/interface.h>
#include <torch/csrc/jit/fuser/common/utils.h>

// Historic fuser includes
#include <torch/csrc/jit/fuser/compiler.h>
#include <torch/csrc/jit/fuser/executor.h>
#include <torch/csrc/jit/fuser/fallback.h>
#include <torch/csrc/jit/fuser/kernel_cache.h>

#include <iterator>

namespace torch {
namespace jit {

using namespace torch::jit::fuser;

namespace {

// TODO: ensure tensors are not sparse (for now)
// TODO: ensure tensors are float tensors (for now)
// Returns true iff:
//  - There is at least one input and one output
//  - All inputs are complete tensors or scalars
//  - All outputs are complete tensors
//  - All tensors are on the same device
bool validateNode(const Node* const node) {
  const auto inputs = node->inputs();
  const auto outputs = node->outputs();

  if (inputs.size() == 0 || outputs.size() == 0) {
    return false;
  }

  const auto device = getFusionDevice(node);

  auto lambda = [device](
    const at::ArrayRef<const Value*> values
  , const bool allow_scalars) {
    for (const auto* const val : values) {
      if (val->isCompleteTensor()) {
        const auto cur_device = *(val->type()->expect<TensorType>()->device());
        if (device != cur_device) {
          return false;
        }
      } else if (!allow_scalars || !isScalar(val)) {
        return false;
      }
    }

    return true;
  };

  return (lambda(inputs, true) && lambda(outputs, false));
}

} // namespace

// Returns true iff the node is fusible
bool isFusible(const Node* const node) {
  if (!validateNode(node)) {
    return false;
  }

  const auto device_type = getFusionDeviceType(node);

  switch (device_type) {
    case c10::kCPU:
      return cpu::isFusibleOnCPU(node);
    case c10::kCUDA:
      return false;
    default:
      return false;
  }

  TORCH_CHECK(false, "End of non-void function");
}

// Returns the key corresponding to the fusion
int fuse(const Node* const node) {
  TORCH_CHECK(isFusible(node), "Asked to create an impossible fusion!");

  const auto device_type = getFusionDeviceType(node);

  switch (device_type) {
    case c10::kCPU:
      return cpu::fuseOnCPU(node);
    case c10::kCUDA:
      return -1;
    default:
      TORCH_CHECK(false, "Trying to fuse on device type that doesn't support fusion!");
  }

  TORCH_CHECK(false, "End of non-void function");
}

void compileFusion(Node* fusion) {
  const auto device_type = getFusionDeviceType(fusion);

  switch (device_type) {
    case c10::kCPU:
      cpu::compileFusionOnCPU(fusion);
      return;
    case c10::kCUDA:
      return;
    default:
      TORCH_CHECK(false, "Trying to fuse on device type that doesn't support fusion!");
  }

  TORCH_CHECK(false, "End of function should not be reached!");
}

// Acquires inputs, allocates outputs, and calls fusion
// TODO: outputs should be preallocated in the graph (see fusion pass)
void callFusion(const Node* const fusion, Stack& stack) {
  // Acquires inputs
  const Graph& graph = *fusion->g(attr::Subgraph);
  const auto nInputs = graph.inputs().size();
  at::ArrayRef<IValue> inputs = last(stack, nInputs);
  drop(stack, nInputs);

  // Constructs output
  std::vector<at::Tensor> outputs;
  for (const auto* const output : graph.outputs()) {
    auto type = output->type()->expect<TensorType>();

    const auto device = *(type->device());
    const auto scalar_type = *(type->scalarType());

    auto options = at::TensorOptions()
      .dtype(scalar_type)
      .layout(at::kStrided)
      .device(device)
      .requires_grad(type->requires_grad());

    const auto sizes = extractSizes(type);

    //auto tensor = at::empty({10, 7, 3, 5}, options);
    auto tensor = at::empty(sizes, options);
    outputs.push_back(tensor);
  }

  // Adds outputs to stack
  stack.insert(
    stack.end()
  , std::make_move_iterator(outputs.begin())
  , std::make_move_iterator(outputs.end()));

  // Calls fusion
  const auto device = *(graph.outputs()[0]->type()->expect<TensorType>()->device());
  switch(device.type()) {
    case c10::kCPU:
      cpu::callFusionOnCPU(fusion, outputs, inputs);
      return;
    case c10::kCUDA:
      return;
    default:
      TORCH_CHECK(false, "Acquired an unknown fusion device type!");
  }
}





// OLD STUFF BELOW HERE



namespace detail {

// Note: CPU fusion is currently disabled due to test flakiness
bool cpu_fuser_enabled = false;

} // namespace detail


int64_t registerFusion(const Node* fusion_group) {
  return fuser::registerFusion(fusion_group);
}

void runFusion(const int64_t key, Stack& stack) {
  const auto result = fuser::runFusion(key, stack);
  if (!result)
    fuser::runFallback(key, stack);
}

bool canFuseOnCPU() {
  return fuser::hasFusionBackend(at::DeviceType::CPU) &&
      detail::cpu_fuser_enabled;
}

bool canFuseOnGPU() {
  return fuser::hasFusionBackend(at::DeviceType::CUDA);
}

void overrideCanFuseOnCPU(bool value) {
  detail::cpu_fuser_enabled = value;
}

// Uses the above interface by stuffing the graph into a node and treating that
// node as a fusion group.
std::vector<at::Tensor> debugLaunchGraph(
    Graph& graph,
    at::ArrayRef<at::Tensor> inputs) {
  // Creates a fusion group node
  auto wrapper_graph = std::make_shared<Graph>();
  Node* fusion_group = wrapper_graph->insertNode(
      wrapper_graph->createWithSubgraph(prim::FusionGroup));
  fusion_group->g_(attr::Subgraph, graph.copy());
  for (size_t i = 0; i < graph.inputs().size(); ++i) {
    fusion_group->addInput(wrapper_graph->addInput());
  }
  for (size_t i = 0; i < graph.outputs().size(); ++i) {
    wrapper_graph->registerOutput(fusion_group->addOutput());
  }

  // Creates the stack, registers and runs the fusion
  Stack stack = fmap<IValue>(inputs);
  const auto key = fuser::registerFusion(fusion_group);
  fuser::runFusion(key, stack);
  return fmap(stack, [](const IValue& iv) { return iv.toTensor(); });
}

std::string debugGetFusedKernelCode(
    Graph& graph,
    at::ArrayRef<at::Tensor> inputs) {
  // Creates a fusion group node
  auto wrapper_graph = std::make_shared<Graph>();
  Node* fusion_group =
      wrapper_graph->insertNode(wrapper_graph->createWithSubgraph(prim::FusionGroup));
  fusion_group->g_(attr::Subgraph, graph.copy());
  for (size_t i = 0; i < graph.inputs().size(); ++i) {
    fusion_group->addInput(wrapper_graph->addInput());
  }
  for (size_t i = 0; i < graph.outputs().size(); ++i) {
    wrapper_graph->registerOutput(fusion_group->addOutput());
  }

  // Creates the stack, registers and runs the fusion
  Stack stack = fmap<IValue>(inputs);
  const auto key = fuser::registerFusion(fusion_group);

  std::string code;
  if (!fuser::runFusion(key, stack, &code)) {
    throw std::runtime_error("Could not run fusion for graph");
  }

  return code;
}

size_t nCompiledKernels() {
  return fuser::nCompiledKernels();
}

} // namespace jit
} // namespace torch

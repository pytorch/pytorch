#include <torch/csrc/jit/codegen/fuser/interface.h>

#include <torch/csrc/jit/codegen/fuser/compiler.h>
#include <torch/csrc/jit/codegen/fuser/executor.h>
#include <torch/csrc/jit/codegen/fuser/fallback.h>
#include <torch/csrc/jit/codegen/fuser/kernel_cache.h>

#include <c10/util/Flags.h>
#include <stdexcept>

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_bool(torch_jit_enable_cpu_fusion, false, "enable cpu fusion");

namespace torch {
namespace jit {

namespace detail {

// Note: CPU fusion is currently disabled due to test flakiness
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
#if defined(FBCODE_CAFFE2)
bool cpu_fuser_enabled = true;
#else
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool cpu_fuser_enabled = false;
#endif

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool gpu_fuser_enabled = true;

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
  return fuser::hasFusionBackend(DeviceType::CPU) &&
      (detail::cpu_fuser_enabled || FLAGS_torch_jit_enable_cpu_fusion);
}

bool canFuseOnGPU() {
  return fuser::hasFusionBackend(DeviceType::CUDA) && detail::gpu_fuser_enabled;
}

void overrideCanFuseOnCPU(bool value) {
  detail::cpu_fuser_enabled = value;
}

void overrideCanFuseOnGPU(bool value) {
  detail::gpu_fuser_enabled = value;
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

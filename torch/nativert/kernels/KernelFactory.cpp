#include <string_view>

#include <c10/util/string_view.h>
#include <fmt/ranges.h>

#include <torch/nativert/executor/DelegateExecutor.h>
#include <torch/nativert/executor/OpKernel.h>
#include <torch/nativert/executor/ParallelGraphExecutor.h>
#include <torch/nativert/executor/SerialGraphExecutor.h>
#include <torch/nativert/graph/Graph.h>
#include <torch/nativert/kernels/AutoFunctionalizeKernel.h>
#include <torch/nativert/kernels/C10Kernel.h>
#include <torch/nativert/kernels/CallTorchBindKernel.h>
#include <torch/nativert/kernels/HigherOrderKernel.h>
#include <torch/nativert/kernels/KernelFactory.h>
#include <torch/nativert/kernels/PrimKernelRegistry.h>

namespace torch::nativert {

inline constexpr std::array<std::string_view, 7> kSymIntOps = {
    "_operator.floordiv",
    "_operator.mod",
    "torch.sym_int",
    "torch.sym_float",
    "torch.sym_ite",
    "torch.sym_max",
    "torch.sym_min",
};

inline constexpr std::array<std::string_view, 8> kSymBoolOps = {
    "_operator.eq",
    "_operator.ne",
    "_operator.le",
    "_operator.ge",
    "_operator.lt",
    "_operator.gt",
    "_operator.and_",
    "torch.sym_not",
};

inline constexpr std::array<std::string_view, 4> kSymFloatOps = {
    "torch._sym_sqrt",
    "math.trunc",
    "_operator.neg",
    "_operator.truediv",
};

inline constexpr std::array<std::string_view, 4> kScalarBinaryOps = {
    "_operator.mul",
    "_operator.add",
    "_operator.sub",
    "_operator.pow",
};

namespace {

struct KernelFactoryRegistry {
  std::unordered_map<std::string, KernelFactoryHandler> handlers;
};

c10::Synchronized<KernelFactoryRegistry>& getKernelFactoryRegistry() {
  static auto* registry = new c10::Synchronized<KernelFactoryRegistry>();
  return *registry;
}

} // namespace

void KernelFactory::registerHandler(
    const std::string& name,
    KernelFactoryHandler handler) {
  auto& registry = getKernelFactoryRegistry();
  registry.withLock([&](auto&& reg) {
    if (reg.handlers.find(name) != reg.handlers.end()) {
      TORCH_CHECK(false, "Handler for ", name, " already registered");
    }
    reg.handlers.emplace(name, std::move(handler));
  });
}

ExecutionKernels KernelFactory::initializeNodeKernels(
    const Graph& graph,
    const std::shared_ptr<Weights>& weights,
    const torch::nativert::ExecutorConfig& executorConfig,
    const std::shared_ptr<caffe2::serialize::PyTorchStreamReader>&
        pytorchStreamReader) {
  std::vector<std::unique_ptr<OpKernel>> nodeKernels;
  std::vector<std::unique_ptr<DelegateExecutor>> delegateExecutors;
  std::vector<ConstFoldingExecution> constFoldingExecutions;

  std::unordered_map<std::string, int> opsWithoutStaticDispatchCount;

  VLOG(1) << fmt::format(
      "PrimKernelRegistry: {}", fmt::join(PrimKernelRegistry()->Keys(), ", "));

  std::unordered_map<std::string, KernelFactoryHandler> handlers;
  getKernelFactoryRegistry().withLock(
      [&](auto&& reg) { handlers = reg.handlers; });

  for (const auto& node : graph.nodes()) {
    std::string target = std::string(node.target());

    bool matched = false;
    for (const auto& [_, handler] : handlers) {
      if (handler.match(node, executorConfig)) {
        auto [kernel, delegate] =
            handler(node, weights, executorConfig, pytorchStreamReader.get());
        if (kernel) {
          nodeKernels.push_back(std::move(kernel));
        }
        if (delegate) {
          delegateExecutors.push_back(std::move(delegate));
        }
        matched = true;
        break;
      }
    }
    if (matched) {
      continue;
    }

    if (PrimKernelRegistry()->Has(target)) {
      nodeKernels.push_back(PrimKernelRegistry()->Create(target, &node));
    } else if (c10::starts_with(
                   node.target(), "torch.ops.higher_order.call_torchbind")) {
      nodeKernels.push_back(std::make_unique<CallTorchBindKernel>(&node));
    } else if (
        c10::starts_with(
            node.target(),
            "torch.ops.higher_order.auto_functionalized") ||
        c10::starts_with( // TODO Remove this condition once the old
                          // pt2 archives are expired.
            node.target(),
            "torch._higher_order_ops.auto_functionalize.auto_functionalized")) {
      nodeKernels.push_back(
          std::make_unique<UnsafeAutoFunctionalizeKernel>(&node));
    } else if (
        std::find(
            std::begin(kSymIntOps), std::end(kSymIntOps), node.target()) !=
        std::end(kSymIntOps)) {
      nodeKernels.push_back(std::make_unique<SymIntOpKernel>(&node));
    } else if (
        std::find(
            std::begin(kSymBoolOps), std::end(kSymBoolOps), node.target()) !=
        std::end(kSymBoolOps)) {
      nodeKernels.push_back(std::make_unique<SymBoolOpKernel>(&node));
    } else if (
        std::find(
            std::begin(kSymFloatOps), std::end(kSymFloatOps), node.target()) !=
        std::end(kSymFloatOps)) {
      nodeKernels.push_back(std::make_unique<SymFloatOpKernel>(&node));
    } else if (
        std::find(
            std::begin(kScalarBinaryOps),
            std::end(kScalarBinaryOps),
            node.target()) != std::end(kScalarBinaryOps)) {
      nodeKernels.push_back(std::make_unique<ScalarBinaryOpKernel>(&node));
    } else if (c10::starts_with(node.target(), "torch.ops.higher_order")) {
      std::vector<std::unique_ptr<GraphExecutorBase>> graphExecutors;
      for (const auto& attr : node.attributes()) {
        if (std::holds_alternative<std::unique_ptr<Graph>>(attr.value)) {
          const auto& subgraph = std::get<std::unique_ptr<Graph>>(attr.value);
          auto executionKernels =
              initializeNodeKernels(*subgraph, weights, executorConfig);
          TORCH_CHECK(
              executionKernels.delegateExecutors.empty(),
              "HigherOrderKernel does not support delegates");
          TORCH_CHECK(
              executionKernels.constFoldingExecutions.empty(),
              "HigherOrderKernel does not support const folding");
          if (executorConfig.maxParallelOps > 1) {
            graphExecutors.emplace_back(
                std::unique_ptr<GraphExecutorBase>(new ParallelGraphExecutor(
                    *subgraph,
                    std::move(executionKernels.nodeKernels),
                    executorConfig)));
          } else {
            graphExecutors.emplace_back(std::unique_ptr<GraphExecutorBase>(
                new torch::nativert::SerialGraphExecutor(
                    *subgraph,
                    std::move(executionKernels.nodeKernels),
                    executorConfig)));
          }
        }
      }
      if (node.target() == "torch.ops.higher_order.run_const_graph") {
        constFoldingExecutions.push_back(
            ConstFoldingExecution{std::move(graphExecutors[0])});
      }
      nodeKernels.push_back(std::make_unique<HigherOrderKernel>(
          &node, std::move(graphExecutors)));
    } else if (c10::starts_with(node.target(), "torch.ops")) {
      nodeKernels.push_back(std::make_unique<C10Kernel>(&node));

      std::string opName = std::string(node.target());
      if (opsWithoutStaticDispatchCount.find(opName) ==
          opsWithoutStaticDispatchCount.end()) {
        opsWithoutStaticDispatchCount[opName] = 0;
      }
      opsWithoutStaticDispatchCount[opName] += 1;
    } else {
      TORCH_CHECK(false, "Unsupported operator: ", target);
    }
  }

  if (executorConfig.enableStaticCPUKernels &&
      !opsWithoutStaticDispatchCount.empty()) {
    std::stringstream ss;
    for (const auto& [op, count] : opsWithoutStaticDispatchCount) {
      ss << op << ": " << count << ", \n";
    }
    LOG(WARNING) << "Following ops are missing static dispatched kernels: \n"
                 << ss.str();
  }

  return {
      std::move(nodeKernels),
      std::move(delegateExecutors),
      std::move(constFoldingExecutions)};
}
} // namespace torch::nativert

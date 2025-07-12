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

namespace {

c10::Device inferTargetDevice(
    const Node& node,
    const std::unordered_map<std::string, torch::nativert::TensorMeta>&
        tensorValuesMeta,
    const Placement& placement) {
  if (node.target() == "prim.Input" || node.target() == "prim.Output") {
    return c10::Device(c10::DeviceType::CPU);
  }

  std::vector<c10::Device> devices;
  for (auto& output : node.outputs()) {
    if (output->type() == Type::Kind::Tensor) {
      auto it = tensorValuesMeta.find(std::string{output->name()});
      if (it != tensorValuesMeta.end()) {
        devices.emplace_back(it->second.device());
      }
    } else if (output->type() == Type::Kind::TensorList) {
      for (const auto& el : output->getListElements()) {
        auto it = tensorValuesMeta.find(std::string{el->name()});
        if (it != tensorValuesMeta.end()) {
          devices.emplace_back(it->second.device());
        }
      }
    }
  }

  if (devices.empty()) {
    return c10::Device(c10::DeviceType::CPU);
  } else {
    for (size_t i = 1; i < devices.size(); ++i) {
      if (!torch::nativert::isSameDevice(devices[0], devices[i])) {
        LOG(WARNING) << "Node " << node
                     << " has outputs on multiple devices: " << devices[0]
                     << " and " << devices[i];
      }
    }

    return placement.getMappedDevice(devices[0]);
  }
}

} // namespace

inline constexpr std::string_view kSymIntOps[] = {
    "_operator.floordiv",
    "_operator.mod",
    "torch.sym_int",
    "torch.sym_float",
    "torch.sym_ite",
    "torch.sym_max",
    "torch.sym_min",
};

inline constexpr std::string_view kSymBoolOps[] = {
    "_operator.eq",
    "_operator.ne",
    "_operator.le",
    "_operator.ge",
    "_operator.lt",
    "_operator.gt",
    "_operator.and_",
    "torch.sym_not",
};

inline constexpr std::string_view kSymFloatOps[] = {
    "torch._sym_sqrt",
    "math.trunc",
    "_operator.neg",
    "_operator.truediv",
};

inline constexpr std::string_view kScalarBinaryOps[] = {
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
    std::shared_ptr<Weights> weights,
    const torch::nativert::ExecutorConfig& executorConfig,
    const Placement& placement,
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader> pytorchStreamReader,
    const MakeProxyExecutorFn& makeProxyExecutorFunc) {
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

    c10::Device targetDevice =
        inferTargetDevice(node, graph.tensorValuesMeta(), placement);

    bool matched = false;
    for (const auto& [_, handler] : handlers) {
      if (handler.match(node, executorConfig, targetDevice)) {
        auto [kernel, delegate] = handler(
            node,
            weights,
            executorConfig,
            pytorchStreamReader.get(),
            targetDevice);
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
          auto executionKernels = initializeNodeKernels(
              *subgraph, weights, executorConfig, placement);
          CHECK(executionKernels.delegateExecutors.empty())
              << "HigherOrderKernel does not support delegates";
          CHECK(executionKernels.constFoldingExecutions.size() == 0)
              << "HigherOrderKernel does not support const folding";
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
      nodeKernels.push_back(std::make_unique<C10Kernel>(&node, targetDevice));

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

  if (executorConfig.enableStaticCPUKernels) {
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

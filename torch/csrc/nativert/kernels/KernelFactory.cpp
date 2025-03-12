#include "torch/csrc/nativert/kernels/KernelFactory.h"
#include <string_view>

#include <c10/util/Logging.h>
#include <fmt/format.h>

#include "torch/csrc/nativert/executor/AOTIDelegateExecutor.h"
#include "torch/csrc/nativert/kernels/AOTICallDelegateKernel.h"
#include "torch/csrc/nativert/kernels/AOTIKernel.h"

#ifdef __SIGRID_USE_FBA__
#include "sigmoid/backend/MTIADelegateExecutor.h"
#include "sigmoid/backend/MTIAKernel.h"
#endif

#include "torch/csrc/nativert/common/String.h"
#include "torch/csrc/nativert/executor/DelegateExecutor.h"
#include "torch/csrc/nativert/executor/OpKernel.h"
#include "torch/csrc/nativert/executor/ParallelGraphExecutor.h"
#include "torch/csrc/nativert/executor/SerialGraphExecutor.h"
#include "torch/csrc/nativert/graph/Graph.h"
#include "torch/csrc/nativert/graph/GraphPasses.h"
#include "torch/csrc/nativert/kernels/AutoFunctionalizeKernel.h"
#include "torch/csrc/nativert/kernels/C10Kernel.h"
#include "torch/csrc/nativert/kernels/CallTorchBindKernel.h"
#include "torch/csrc/nativert/kernels/HigherOrderKernel.h"
#include "torch/csrc/nativert/kernels/KernelRegistry.h"

namespace torch::nativert {

namespace {

c10::Device inferTargetDevice(
    const Node& node,
    const std::unordered_map<std::string, TensorMeta>& tensorValuesMeta,
    const Placement& placement) {
  if (node.target() == "prim.Input" || node.target() == "prim.Output") {
    return c10::Device(c10::DeviceType::CPU);
  }

  std::vector<c10::Device> devices;
  for (auto& output : node.outputs()) {
    if (output->type() == Type::Tensor) {
      auto it = tensorValuesMeta.find(std::string{output->name()});
      if (it != tensorValuesMeta.end()) {
        devices.emplace_back(it->second.device());
      }
    } else if (output->type() == Type::TensorList) {
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
      if (!isSameDevice(devices[0], devices[i])) {
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
};

inline constexpr std::string_view kScalarBinaryOps[] = {
    "_operator.mul",
    "_operator.add",
    "_operator.sub",
    "_operator.pow",
};

namespace {

const std::string maybeRevisedStaticDispatchTarget(const Node& node) {
  auto overloadName = selectScalarOverloadName(node);
  if (!ends_with(node.target(), overloadName)) {
    const std::string& newTarget =
        std::string(node.target())
            .replace(node.target().rfind('.'), std::string::npos, overloadName);
    LOG(INFO) << fmt::format(
        "Converting Tensor to {} for node: {} -> {}",
        overloadName,
        node.target(),
        newTarget);
    return newTarget;
  }
  return std::string(node.target());
}

} // namespace

ExecutionKernels KernelFactory::initializeNodeKernels(
    const Graph& graph,
    std::shared_ptr<Weights> weights,
    const ExecutorConfig& executorConfig,
    const Placement& placement,
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader>
        pytorchStreamReader) {
  std::vector<std::unique_ptr<OpKernel>> nodeKernels;
  std::vector<std::unique_ptr<DelegateExecutor>> delegateExecutors;
  std::vector<ConstFoldingExecution> constFoldingExecutions;

  std::unordered_set<std::string> opsWithoutStaticDispatch;

  VLOG(1) << "PrimKernelRegistry: " << join(", ", PrimKernelRegistry()->Keys());
  VLOG(1) << "StaticallyDispatchedCPUKernelRegistry: "
          << join(", ", StaticallyDispatchedCPUKernelRegistry()->Keys());

  for (const auto& node : graph.nodes()) {
    std::string originalTarget = std::string(node.target());
    const std::string target =
        (executorConfig.enableStaticCPUKernels &&
         StaticallyDispatchedCPUKernelRegistry()->Has(originalTarget))
        ? maybeRevisedStaticDispatchTarget(node)
        : std::move(originalTarget);
    c10::Device targetDevice =
        inferTargetDevice(node, graph.tensorValuesMeta(), placement);

    if (PrimKernelRegistry()->Has(target)) {
      nodeKernels.push_back(PrimKernelRegistry()->Create(target, &node));
    } else if (
        executorConfig.enableStaticCPUKernels &&
        StaticallyDispatchedCPUKernelRegistry()->Has(target) &&
        targetDevice.is_cpu()) {
      nodeKernels.push_back(StaticallyDispatchedCPUKernelRegistry()->Create(
          target, &node, targetDevice));
    } else if (starts_with(
                   node.target(), "torch.ops.higher_order.call_torchbind")) {
      nodeKernels.push_back(std::make_unique<CallTorchBindKernel>(&node));
    } else if (
        starts_with(
            node.target(),
            "torch.ops.higher_order.auto_functionalized") ||
        starts_with( // TODO Remove this condition once the old
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
    } else if (starts_with(
                   node.target(), "torch.ops.delegate.call_aotinductor")) {
      const auto pathAttr = node.tryGetAttribute("path");
      CHECK(pathAttr != nullptr);

      const Constant& pathValue = pathAttr->value;
      CHECK(std::holds_alternative<std::string>(pathValue));
      std::string path = std::get<std::string>(pathValue);

      auto delegateExecutor = std::make_unique<AOTIDelegateExecutor>(
          path, weights, targetDevice, executorConfig, pytorchStreamReader);
      nodeKernels.push_back(
          std::make_unique<AOTIKernel>(&node, *delegateExecutor));
      delegateExecutors.push_back(std::move(delegateExecutor));
    } else if (starts_with(
                   node.target(),
                   "torch.ops.higher_order.aoti_call_delegate")) {
      // the first attribute is serialized as the path to the aotinductor
      const auto pathAttr = node.attributes().begin();
      const Constant& pathValue = pathAttr->value;
      CHECK(std::holds_alternative<std::string>(pathValue));
      std::string path = std::get<std::string>(pathValue);

      auto delegateExecutor = std::make_unique<AOTIDelegateExecutor>(
          path, weights, targetDevice, executorConfig, pytorchStreamReader);
      nodeKernels.push_back(
          std::make_unique<AOTICallDelegateKernel>(&node, *delegateExecutor));
      delegateExecutors.push_back(std::move(delegateExecutor));
    } else if (starts_with(node.target(), "torch.ops.delegate.call_mtia")) {
#ifdef __SIGRID_USE_FBA__
      auto delegateExecutor = std::make_unique<MTIADelegateExecutor>(
          &node, weights, executorConfig, pytorchStreamReader);
      nodeKernels.push_back(
          std::make_unique<MTIAKernel>(&node, *delegateExecutor));
      delegateExecutors.push_back(std::move(delegateExecutor));
#endif
    } else if (starts_with(node.target(), "torch.ops.higher_order")) {
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
            graphExecutors.emplace_back(
                std::unique_ptr<GraphExecutorBase>(new SerialGraphExecutor(
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
    } else if (starts_with(node.target(), "torch.ops")) {
      nodeKernels.push_back(std::make_unique<C10Kernel>(&node, targetDevice));

      opsWithoutStaticDispatch.insert(std::string(node.target()));
    } else {
      TORCH_CHECK(false, "Unsupported operator: ", target);
    }
  }

  if (executorConfig.enableStaticCPUKernels) {
    std::stringstream ss;
    for (const auto& op : opsWithoutStaticDispatch) {
      ss << op << ", ";
    }
    LOG(WARNING) << "Following ops are missing static dispatched kernels: "
                 << ss.str();
  }

  return {
      std::move(nodeKernels),
      std::move(delegateExecutors),
      std::move(constFoldingExecutions)};
}
} // namespace torch::nativert

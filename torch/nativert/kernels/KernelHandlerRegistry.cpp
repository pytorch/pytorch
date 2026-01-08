#include <torch/nativert/kernels/KernelHandlerRegistry.h>

#include <c10/util/Logging.h>
#include <fmt/format.h>

#include <ATen/core/ivalue.h>
#include <c10/util/CallOnce.h>

#include <torch/nativert/graph/Graph.h>
#include <torch/nativert/graph/GraphPasses.h>
#include <torch/nativert/graph/GraphUtils.h>
#include <torch/nativert/kernels/KernelFactory.h>
#include <torch/nativert/kernels/KernelRegistry.h>

#include <torch/csrc/inductor/aoti_torch/oss_proxy_executor.h>
#include <torch/nativert/executor/AOTInductorDelegateExecutor.h>
#include <torch/nativert/kernels/ETCallDelegateKernel.h>

namespace torch::nativert {

namespace {
std::string maybeRevisedStaticDispatchTarget(const Node& node) {
  auto overloadName = selectScalarOverloadName(node);

  if (!overloadName.empty() && !c10::ends_with(node.target(), overloadName)) {
    const std::string newTarget =
        std::string(node.target())
            .replace(
                node.target().rfind('.') + 1, std::string::npos, overloadName);
    LOG(INFO) << fmt::format(
        "Converting Tensor to {} for node: {} -> {}",
        overloadName,
        node.target(),
        newTarget);
    return newTarget;
  }
  return std::string(node.target());
}

void updateNodeTargetIfNeeded(Node& node) {
  auto newTarget = maybeRevisedStaticDispatchTarget(node);
  node.setTarget(newTarget);
}

std::unique_ptr<torch::aot_inductor::ProxyExecutor> make_proxy_executor(
    const std::string& filename,
    bool is_cpu,
    std::optional<std::unordered_map<std::string, c10::IValue>> custom_objs) {
  return std::make_unique<torch::aot_inductor::OSSProxyExecutor>(
      filename, is_cpu, std::move(custom_objs));
}
} // namespace

void register_kernel_handlers() {
  static c10::once_flag flag;
  c10::call_once(flag, []() {
    using OpKernelPtr = KernelFactoryHandler::OpKernelPtr;
    using DelegateExecutorPtr = KernelFactoryHandler::DelegateExecutorPtr;
    KernelFactory::registerHandler(
        "static_cpu",
        KernelFactoryHandler(
            [](const Node& node,
               const torch::nativert::ExecutorConfig& executorConfig) {
              if (!executorConfig.enableStaticCPUKernels ||
                  !torch::nativert::areAllIOTensorsAttributesOnCpu(node)) {
                return false;
              }
              const std::string target = maybeRevisedStaticDispatchTarget(node);
              return torch::nativert::StaticallyDispatchedCPUKernelRegistry()
                  ->Has(target);
            },
            [](const Node& node,
               // NOLINTNEXTLINE(performance-unnecessary-value-param)
               std::shared_ptr<Weights> weights,
               const torch::nativert::ExecutorConfig& executorConfig,
               caffe2::serialize::PyTorchStreamReader* packageReader)
                -> std::pair<OpKernelPtr, DelegateExecutorPtr> {
              updateNodeTargetIfNeeded(const_cast<Node&>(node));

              return {
                  torch::nativert::StaticallyDispatchedCPUKernelRegistry()
                      ->Create(maybeRevisedStaticDispatchTarget(node), &node),
                  nullptr};
            }));
    KernelFactory::registerHandler(
        "et_delegate",
        KernelFactoryHandler(
            [](const Node& node,
               const torch::nativert::ExecutorConfig& /* executorConfig */) {
              return c10::starts_with(
                  node.target(),
                  "torch.ops.higher_order.executorch_call_delegate");
            },
            [](const Node& node,
               // NOLINTNEXTLINE(performance-unnecessary-value-param)
               std::shared_ptr<Weights> weights,
               const torch::nativert::ExecutorConfig& executorConfig,
               caffe2::serialize::PyTorchStreamReader* packageReader)
                -> std::pair<
                    KernelFactoryHandler::OpKernelPtr,
                    KernelFactoryHandler::DelegateExecutorPtr> {
              auto delegateExecutor = std::make_unique<AOTIDelegateExecutor>(
                  node,
                  weights,
                  executorConfig,
                  packageReader,
                  make_proxy_executor);

              return {
                  std::make_unique<ETCallDelegateKernel>(
                      &node, *delegateExecutor),
                  std::move(delegateExecutor)};
            }));
  });
}

} // namespace torch::nativert



#pragma once

#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "torch/csrc/nativert/common/Pytree.h"
#include "torch/csrc/nativert/executor/Executor.h"
#include "torch/csrc/nativert/executor/Placement.h"

namespace torch::nativert {

using ExtraFilesMap = std::unordered_map<std::string, std::string>;

enum class ExecutorType {
  INTERPRETER = 0,
  AOTINDUCTOR = 1,
  MTIA = 2,
};

struct BaseRuntimeConfigs {
  bool isDebug = false;

  bool validateInputs = false;

  // use static kernels
  bool enableStaticCPUKernels = false;

  // whether to enable static memory planning
  bool enableStaticMemoryPlanning = false;

  // whether to load node's metadata, e.g. stacktrace etc.
  // This is only used for debugging purpose. For production, we commonly set
  // this to false, as it would incur extra memory usage.
  bool loadNodeMetadata = false;

  // whether to initialize the executor in the constructor
  // In some cases, Weights are not available when the constructor is called.
  // In this case, the executor should be initialized later.
  bool deferInitialization = false;

  // platform arch for delegate, e.g "cpu", "sm80_x86" etc
  // See https://fburl.com/code/3pym9ipj for supported platforms
  std::string platformArch;

  // allows up to max number of concurrent threads.
  int64_t maxNumConcurrentThreads = 8;

  // allows up to max number of parallel ops.
  int64_t maxParallelOps = 1;

  // whether to enable runtime const folding
  bool enableRuntimeConstFolding = false;
};

struct RunConfigs {};

class TORCH_API ModelRunnerBase {
 public:
  ModelRunnerBase(
      std::shared_ptr<caffe2::serialize::PyTorchStreamReader>
          pytorchStreamReader,
      const std::string& modelName,
      ExecutorType executorType,
      const BaseRuntimeConfigs& runtimeConfigs,
      // functor to build the placement after the graph is loaded, but before
      // loading the weights.
      const std::function<Placement(const torch::nativert::Graph& graph)>&
          buildPlacementFn);

  ModelRunnerBase(ModelRunnerBase&&) = default;
  ModelRunnerBase& operator=(ModelRunnerBase&&) = default;

  ModelRunnerBase(const ModelRunnerBase&) = delete;
  ModelRunnerBase& operator=(const ModelRunnerBase&) = delete;

  virtual ~ModelRunnerBase() = default;

  const std::string& getModelName() const;

  std::shared_ptr<Weights> getWeights();

  // loadNewWeights() loads the weights from the specified model into
  // the newWeights_ buffer. The weights will stay shadow and should be
  // actualized by the commitNewWeights() function.
  std::shared_ptr<Weights> loadNewWeights(
      std::shared_ptr<caffe2::serialize::PyTorchStreamReader>
          packageStreamReader,
      std::function<bool(const std::string&)> skipSizeCheck = {},
      std::function<bool(const std::string&)> skipDtypeCheck = {});

  void commitNewWeights();

  c10::IValue run(
      const std::unordered_map<std::string, c10::IValue>& kwargs,
      const RunConfigs& runConfigs = RunConfigs());

  c10::IValue run(
      const std::vector<c10::IValue>& args,
      const std::unordered_map<std::string, c10::IValue>& kwargs,
      const RunConfigs& runConfigs = RunConfigs());

  /**
   * A low level API which expects user to always pass in flattened inputs.
   * The ownership of the entire input list must be transferred to the
   * executor via std::move or in-place construction.
   */
  std::vector<c10::IValue> runWithFlatInputsAndOutputs(
      std::vector<c10::IValue>&& flatInputs,
      const RunConfigs& runConfigs = RunConfigs());

  void benchmarkIndividualNodes(
      const std::vector<std::vector<c10::IValue>>& argsList,
      const std::vector<std::unordered_map<std::string, c10::IValue>>&
          kwargsList,
      const uint32_t warmupRuns,
      const uint32_t mainRuns,
      const bool printPerNodeTime,
      const RunConfigs& runConfigs);

  std::vector<std::optional<c10::Device>> getUserInputTargetDevices() const;

  std::pair<
      std::vector<c10::IValue>,
      std::unordered_map<std::string, c10::IValue>>
  loadSampleInputs(
      std::shared_ptr<caffe2::serialize::PyTorchStreamReader>
          pytorchStreamReader,
      const Placement& placement = Placement());

  const std::vector<std::string>& getArgumentNames() const;
  c10::ArrayRef<const Value*> getArguments() const;

  /*
   * Load extra files indicated in extraFiles from the model package.
   * Return true if any extra files were loaded
   * and false otherwise
   */
  bool loadExtraFiles(
      ExtraFilesMap& extraFiles,
      std::shared_ptr<caffe2::serialize::PyTorchStreamReader>
          pytorchStreamReader);

  const TreeSpec& getOutputSpec() const;
  const TreeSpec& getInputSpec() const;

  void setExecutorType(ExecutorType type, const std::string& platformArch = "");

  ExecutorType getExecutorType() const {
    return executorType_;
  }

  void setEnableStaticDispatchKernels(bool enabled) {
    runtimeConfigs_.enableStaticCPUKernels = enabled;
  }

  void setEnableStaticMemoryPlanning(bool enabled) {
    runtimeConfigs_.enableStaticMemoryPlanning = enabled;
  }

  template <typename T>
  std::vector<T*> getDelegates() {
    std::vector<T*> delegates;
    for (const auto& delegate : executor_->getDelegates()) {
      if (auto* d = dynamic_cast<T*>(delegate)) {
        delegates.push_back(d);
      }
    }
    return delegates;
  }

  // Manually initialize the executor when config.deferInitialization is True.
  //
  // initlaize() must be call after
  // - weights are fully loaded
  // - executor is selected via setExecutorType()
  // ModelRunner is not ready to serve with run() until initlaized.
  //
  // Note that pytorchStreamReader is required to load lowered modules.
  //
  // When initialization failed at any point, it will throw an exception. Caller
  // should catch the exception and call initialize() again with valid weights
  // and executor type.
  //
  // ModelRunner can only be initialized once, it will be noop to call this
  // function when ModelRunner is already ready to serve.
  void initialize(
      std::shared_ptr<caffe2::serialize::PyTorchStreamReader>
          pytorchStreamReader = nullptr);

 protected:
#ifdef ModelRunnerTest_TEST_FRIENDS
  ModelRunnerTest_TEST_FRIENDS;
#endif

  virtual std::unique_ptr<Graph> deserializeDelegateGraph() const = 0;
  void initializeExecutor(
      std::shared_ptr<Weights> weights,
      std::shared_ptr<caffe2::serialize::PyTorchStreamReader>
          pytorchStreamReader);

  std::string loadSerializedModel(
      std::shared_ptr<caffe2::serialize::PyTorchStreamReader>
          pytorchStreamReader) const;

  std::string modelName_;
  ExecutorType executorType_;
  BaseRuntimeConfigs runtimeConfigs_;

  Placement placement_;

  // original non-delegated graph from torch.export()
  std::shared_ptr<Graph> graph_;

  // graph with delegated nodes after lowering/compilation
  std::shared_ptr<Graph> delegateGraph_;

  // key is weight name, value is archive path for weight
  std::unordered_map<std::string, std::string> stateDictPath_;

  // contains both tensor constants and CustomClassHolder (aka. torchbind
  // object)
  std::unordered_map<std::string, std::string> constantPaths_;

  std::unique_ptr<Executor> executor_;

  // recently loaded and not yet committed weights.
  std::shared_ptr<Weights> newWeights_;

  TreeSpec inputSpec_;
  TreeSpec outputSpec_;
};

} // namespace torch::nativert

template <>
struct fmt::formatter<torch::nativert::ExecutorType> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(torch::nativert::ExecutorType et, FormatContext& ctx) const {
    using namespace torch::nativert;
    switch (et) {
      case ExecutorType::INTERPRETER:
        return format_to(ctx.out(), "INTERPRETER");
      case ExecutorType::AOTINDUCTOR:
        return format_to(ctx.out(), "AOTINDUCTOR");
      case ExecutorType::MTIA:
        return format_to(ctx.out(), "MTIA");
      default:
        return format_to(ctx.out(), "UNKNOWN");
    }
  }
};

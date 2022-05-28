#import <torch/csrc/jit/backends/backend.h>
#import <torch/csrc/jit/backends/coreml/cpp/context.h>
#import <torch/csrc/jit/backends/coreml/objc/PTMCoreMLExecutor.h>
#import <torch/csrc/jit/backends/coreml/objc/PTMCoreMLCompiler.h>
#import <torch/csrc/jit/backends/coreml/objc/PTMCoreMLConfig.h>
#import <torch/csrc/jit/backends/coreml/objc/PTMTensorSpec.h>
#import <torch/csrc/jit/backends/coreml/objc/PTMCoreMLFeatureProvider.h>
#import <torch/csrc/jit/backends/coreml/observer/PTMCoreMLObserver.h>

#import <torch/script.h>
#import <CoreML/CoreML.h>

#if C10_IOS
#import <UIKit/UIKit.h>
#elif TARGET_OS_MAC
#import <Foundation/NSProcessInfo.h>
#endif

namespace torch {
namespace jit {
namespace mobile {
namespace coreml {

static const int32_t kSampleThreshold = static_cast<int32_t>(1.0 / 1000.0 * static_cast<double>(RAND_MAX));
static const int32_t kSampleEvery = 500;

static inline c10::ScalarType scalarType(TensorType type) {
  switch (type) {
    case TensorType::Float:
      return c10::ScalarType::Float;
    case TensorType::Double:
      return c10::ScalarType::Double;
    case TensorType::Int:
      return c10::ScalarType::Int;
    case TensorType::Long:
      return c10::ScalarType::Long;
    case TensorType::Undefined:
      return c10::ScalarType::Undefined;
    default:
      return c10::ScalarType::Undefined;
  }
}

// Wrap the MLModel into a struct to be able to pack into IValue
struct MLModelWrapper: public CustomClassHolder {
 public:
  MLModelWrapper(
    MLModel* mlModel,
    std::vector<TensorSpec>& inputs,
    std::vector<TensorSpec>& outputs,
    int64_t coreMLVersion,
    int32_t modelLoadID,
    size_t initialMemLimit)
    : mlModel(mlModel),
      inputs(inputs),
      outputs(outputs),
      coreMLVersion(coreMLVersion),
      modelLoadID(modelLoadID),
      initialMemLimit(initialMemLimit) {}
  MLModel *mlModel = nullptr;
  std::vector<TensorSpec> inputs;
  std::vector<TensorSpec> outputs;
  int64_t coreMLVersion;
  int32_t modelLoadID;
  int32_t inferences = 0;
  size_t initialMemLimit;
};

class CoreMLBackend: public torch::jit::PyTorchBackendInterface {

 public:
  c10::impl::GenericDict compile(c10::IValue processed, c10::impl::GenericDict method_compile_spec) override {
    auto modelDict = processed.toGenericDict();
    const char *specs = modelDict.at("extra").toStringRef().c_str();
    const std::string& model = modelDict.at("model").toStringRef();
    const std::string& sha256 = modelDict.at("hash").toStringRef();

    int32_t modelLoadID = std::rand();
    int32_t instanceKey = std::rand();
    size_t initialMemLimit = 0;

    PTMCoreMLObserver *observer = coreMLObserverConfig().getCoreMLObserver();
    if (observer) {
      initialMemLimit = observer->getRemainingMemory();
      observer->onEnterCompileModel(instanceKey, modelLoadID);
    }

    NSError *compileError;
    NSDictionary *dict = [PTMCoreMLCompiler parseSpecs:specs error:&compileError];
    if (!dict || compileError) {
      if (observer) {
        observer->onExitCompileModel(instanceKey, false, true);
      }
      TORCH_CHECK(false, "Parsing spec json failed!");
    }

    CoreMLConfig config = CoreMLConfig(dict[@"config"]);
    int64_t coreMLVersion = config.coreMLVersion();
    NSString *backend = config.backend();
    BOOL allowLowPrecision = config.allowLowPrecision();

    MLModel *compiledModel =
      [PTMCoreMLCompiler
       compileMLModel:model
       identifier:sha256
       backend:backend
       allowLowPrecision:allowLowPrecision
       error:&compileError];

    if (!compiledModel || compileError) {
      if (observer) {
        observer->onExitCompileModel(instanceKey, false, true);
      }
      TORCH_CHECK(false, "Compiling MLModel failed!");
    }

    if (observer) {
      bool shouldLog = modelLoadID < kSampleThreshold;
      observer->onExitCompileModel(instanceKey, true, shouldLog);
    }

    NSArray<NSArray *> *inputs = dict[@"inputs"];
    NSArray<NSArray *> *outputs = dict[@"outputs"];
    std::vector<TensorSpec> inputSpecs, outputSpecs;
    for (NSArray *input in inputs) {
      inputSpecs.emplace_back(TensorSpec(input));
    }
    for (NSArray *output in outputs) {
      outputSpecs.emplace_back(TensorSpec(output));
    }

    auto modelWrapper =
      c10::make_intrusive<MLModelWrapper>(
        compiledModel,
        inputSpecs,
        outputSpecs,
        coreMLVersion,
        modelLoadID,
        initialMemLimit);

    auto handle = IValue::make_capsule(modelWrapper);
    c10::Dict<IValue, IValue> ret(StringType::get(), c10::AnyType::get());
    ret.insert("forward", handle);
    return c10::impl::toGenericDict(ret);
  }

  c10::impl::GenericList execute(c10::IValue handle, c10::impl::GenericList inputs) override {
    int32_t instanceKey = std::rand();

    auto modelWrapper = c10::static_intrusive_pointer_cast<MLModelWrapper>(handle.toCapsule());

    PTMCoreMLObserver *observer = coreMLObserverConfig().getCoreMLObserver();
    if (observer) {
      observer->onEnterExecuteModel(
        instanceKey,
        modelWrapper->modelLoadID,
        modelWrapper->initialMemLimit,
        modelWrapper->inferences);
    }

    std::vector<PTMCoreMLFeatureSpecs> inputSpecs;
    int inputSpecIndex = 0;
    // pack the inputs
    for (int i = 0; i < inputs.size(); ++i) {
      auto val = inputs.get(i);
      if (val.isTuple()) {
        auto& tuples = val.toTupleRef().elements();
        for (auto& ival : tuples) {
          TORCH_CHECK(ival.isTensor());
          auto tensor = ival.toTensor();
          PTMCoreMLFeatureSpecs spec{
              .name = modelWrapper->inputs[inputSpecIndex].name(),
              .tensor = tensor,
          };
          inputSpecs.emplace_back(spec);
          ++inputSpecIndex;
        }
      } else {
        TORCH_CHECK(val.isTensor());
        auto tensor = val.toTensor();
        PTMCoreMLFeatureSpecs spec{
            .name = modelWrapper->inputs[inputSpecIndex].name(),
            .tensor = tensor,
        };
        inputSpecs.emplace_back(spec);
        ++inputSpecIndex;
      }
    }

    NSError *error;
    auto outputFeature = [PTMCoreMLExecutor forward:inputSpecs model:modelWrapper->mlModel coreMLVersion:modelWrapper->coreMLVersion error:&error];

    modelWrapper->inferences++;

    if (observer) {
      if (!outputFeature || error) {
        observer->onExitExecuteModel(instanceKey, modelWrapper->inferences, false, true);
      } else {
        // Check if this inference session is being logged.
        // If so, only log every N inferences
        bool shouldLog = modelWrapper->modelLoadID < kSampleThreshold && modelWrapper->inferences > 1;
        if (shouldLog) {
          shouldLog = modelWrapper->inferences % kSampleEvery == 0;
        }
        observer->onExitExecuteModel(instanceKey, modelWrapper->inferences, true, shouldLog);
      }
    }

    // pack the outputs
    c10::List<torch::Tensor> outputs;
    for (auto& spec : modelWrapper->outputs) {
      MLFeatureValue* val = [outputFeature featureValueForName:spec.name()];
      TORCH_CHECK(val.multiArrayValue);
      // Currently, only Float type is supported
      TORCH_CHECK(val.multiArrayValue.dataType == MLMultiArrayDataTypeFloat32);
      std::vector<int64_t> outputShape;
      for (int i = 0; i < val.multiArrayValue.shape.count; ++i) {
        outputShape.emplace_back(val.multiArrayValue.shape[i].integerValue);
      }
      auto tensor =
          at::empty(IntArrayRef(outputShape), scalarType(spec.dtype()));
      int64_t count = val.multiArrayValue.count;
      memcpy(
          tensor.data_ptr<float>(),
          (float*)val.multiArrayValue.dataPointer,
          count * sizeof(float));
      outputs.push_back(tensor);
    }

    return c10::impl::toList(outputs);
  }

  bool is_available() override {
#if !defined(__APPLE__)
    return false;
#elif TARGET_OS_IPHONE
    return [UIDevice currentDevice].systemVersion.floatValue > 14.0;
#elif TARGET_OS_MAC
    NSOperatingSystemVersion supportedVer = {10, 13, 0};
    return [[NSProcessInfo processInfo] isOperatingSystemAtLeastVersion:supportedVer];
#endif
    return false;
  }
};

API_AVAILABLE(ios(11.0), macos(10.13))
static auto cls = torch::jit::backend<CoreMLBackend>("coreml");

} // namespace
}
}
}

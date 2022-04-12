#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/coreml/cpp/context.h>
#include <torch/csrc/jit/backends/coreml/objc/PTMCoreMLExecutor.h>
#include <torch/script.h>

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

static constexpr int SUPPORTED_COREML_VER = 4;

enum TensorType {
  Float,
  Double,
  Int,
  Long,
  Undefined,
};

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

static id parse(NSString* jsonStr) {
  NSData* data = [jsonStr dataUsingEncoding:NSUTF8StringEncoding];
  NSError* error = nil;
  id result = [NSJSONSerialization JSONObjectWithData:data
                                              options:0
                                                error:&error];
  if (error || !result) {
    TORCH_CHECK(
        false,
        "parsing JSON string failed!",
        error.localizedDescription.UTF8String);
  }

  return result;
}

struct TensorSpec {
 public:
  TensorSpec() = delete;
  TensorSpec(NSArray<NSString*>* spec) {
    TORCH_CHECK(spec.count == 3);
    name_ = spec[0];
    dtype_ = (TensorType)spec[1].intValue;
  }
  NSString* name() {
    return name_;
  }
  TensorType dtype() {
    return dtype_;
  }

 private:
  NSString* name_ = @"";
  TensorType dtype_ = TensorType::Float;
};

struct CoreMLConfig {
 public:
  CoreMLConfig() = delete;
  CoreMLConfig(NSDictionary* dict)
      : coreMLVersion_([dict[@"spec_ver"] intValue]),
        backend_([dict[@"backend"] lowercaseString]),
        allow_low_precision_([dict[@"allow_low_precision"] boolValue]) {
    TORCH_CHECK(
        coreMLVersion_ >= SUPPORTED_COREML_VER,
        "Only Core ML version 4 or above are supported");
  }
  int64_t coreMLVersion() const {
    return coreMLVersion_;
  }
  NSString* backend() const {
    return backend_;
  }
  bool allowLowPrecision() const {
    return allow_low_precision_;
  }

 private:
  int64_t coreMLVersion_ = SUPPORTED_COREML_VER;
  NSString* backend_ = @"CPU";
  bool allow_low_precision_ = true;
};

struct MetaData {
 public:
  MetaData(NSDictionary* dict)
      : torchVer_(dict[@"torch_ver"]),
        coremltoolVer_(dict[@"coremltool_ver"]) {}
  NSString* torchVer() const {
    return torchVer_;
  }
  NSString* coremltoolVer() const {
    return coremltoolVer_;
  }

 private:
  NSString* torchVer_ = @"";
  NSString* coremltoolVer_ = @"";
};

// Wrap the Objective-C executor into a C++ to be able to pack into IValue
struct API_AVAILABLE(ios(11.0), macos(10.13)) CoreMLExecutorWrapper
    : public CustomClassHolder {
 public:
  CoreMLExecutorWrapper(
      PTMCoreMLExecutor* executor,
      std::vector<TensorSpec>& inputs,
      std::vector<TensorSpec>& outputs,
      CoreMLConfig config)
      : executor_(executor),
        inputs_(inputs),
        outputs_(outputs),
        config_(config) {}
  c10::List<torch::Tensor> execute(const c10::impl::GenericList& inputs) {
    std::vector<PTMCoreMLFeatureSpecs> inputSpecs;
    std::vector<PTMCoreMLFeatureSpecs> outputSpecs;
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
              .name = inputs_[inputSpecIndex].name(),
              .tensor = tensor,
          };
          inputSpecs.emplace_back(spec);
          ++inputSpecIndex;
        }
      } else {
        TORCH_CHECK(val.isTensor());
        auto tensor = val.toTensor();
        PTMCoreMLFeatureSpecs spec{
            .name = inputs_[inputSpecIndex].name(),
            .tensor = tensor,
        };
        inputSpecs.emplace_back(spec);
        ++inputSpecIndex;
      }
    }
    // pack the outputs
    c10::List<torch::Tensor> outputs;
    id<MLFeatureProvider> results = [executor_ forwardWithInputs:inputSpecs];
    for (auto& spec : outputs_) {
      MLFeatureValue* val = [results featureValueForName:spec.name()];
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
    return outputs;
  }

 private:
  PTMCoreMLExecutor* executor_ = nullptr;
  std::vector<TensorSpec> inputs_;
  std::vector<TensorSpec> outputs_;
  CoreMLConfig config_;
};

class API_AVAILABLE(ios(11.0), macos(10.13)) CoreMLBackend
    : public torch::jit::PyTorchBackendInterface {
 public:
  c10::impl::GenericDict compile(
      c10::IValue processed,
      c10::impl::GenericDict method_compile_spec) override {
    auto modelDict = processed.toGenericDict();
    NSString* specs = [[NSString alloc]
        initWithCString:modelDict.at("extra").toStringRef().c_str()
               encoding:NSUTF8StringEncoding];
    NSDictionary* dict = parse(specs);
    NSArray<NSArray*>* inputs = dict[@"inputs"];
    NSArray<NSArray*>* outputs = dict[@"outputs"];
    std::vector<TensorSpec> inputSpecs, outputSpecs;
    for (NSArray* input in inputs) {
      inputSpecs.emplace_back(TensorSpec(input));
    }
    for (NSArray* output in outputs) {
      outputSpecs.emplace_back(TensorSpec(output));
    }
    auto config = CoreMLConfig(dict[@"config"]);
    const std::string& model = modelDict.at("model").toStringRef();
    const std::string& sha256 = modelDict.at("hash").toStringRef();
    PTMCoreMLExecutor* executor = [PTMCoreMLExecutor new];
    executor.backend = config.backend();
    executor.allowLowPrecision = config.allowLowPrecision();
    executor.coreMLVersion = config.coreMLVersion();
    bool result = [executor compileMLModel:model identifier:sha256];
    TORCH_CHECK(result, "Compiling MLModel failed!");
    auto executorWrapper = c10::make_intrusive<CoreMLExecutorWrapper>(
        executor, inputSpecs, outputSpecs, config);
    auto handle = IValue::make_capsule(executorWrapper);
    c10::Dict<IValue, IValue> ret(StringType::get(), c10::AnyType::get());
    ret.insert("forward", handle);
    return c10::impl::toGenericDict(ret);
  }

  c10::impl::GenericList execute(
      c10::IValue handle,
      c10::impl::GenericList inputs) override {
    auto executor = c10::static_intrusive_pointer_cast<CoreMLExecutorWrapper>(
        handle.toCapsule());
    auto outputs = executor->execute(inputs);
    return c10::impl::toList(outputs);
  }
  bool is_available() override {
    return [PTMCoreMLExecutor isAvailable];
  }
};

struct API_AVAILABLE(ios(11.0), macos(10.13)) ContextImpl
    : public ContextInterface {
  bool isCoreMLAvailable() const override {
    return [PTMCoreMLExecutor isAvailable];
  }
  void setModelCacheDirectory(std::string dir) override {
    [PTMCoreMLExecutor
        setModelCacheDirectory:[NSString stringWithCString:dir.c_str()]];
  }
};

API_AVAILABLE(ios(11.0), macos(10.13))
static auto cls = torch::jit::backend<CoreMLBackend>("coreml");

API_AVAILABLE(ios(11.0), macos(10.13))
static BackendRegistrar g_coreml_backend(new ContextImpl());

} // namespace
}
}
}

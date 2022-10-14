#import <torch/csrc/jit/backends/backend.h>
#import <torch/csrc/jit/backends/coreml/cpp/context.h>
#import <torch/csrc/jit/backends/coreml/objc/PTMCoreMLCompiler.h>
#import <torch/csrc/jit/backends/coreml/objc/PTMCoreMLExecutor.h>
#import <torch/csrc/jit/backends/coreml/objc/PTMCoreMLModelWrapper.h>
#import <torch/csrc/jit/backends/coreml/objc/PTMCoreMLTensorSpec.h>
#import <torch/script.h>

#import <CoreML/CoreML.h>

#if C10_IOS
#import <UIKit/UIKit.h>
#elif TARGET_OS_MAC
#import <Foundation/NSProcessInfo.h>
#endif

// This is a utility macro that can be used to throw an exception when a CoreML
// API function produces a NSError. The exception will contain a message with
// useful info extracted from the NSError.
#define COREML_THROW_IF_ERROR(error, preamble)                                   \
  do {                                                                           \
    if C10_LIKELY(error) {                                                       \
      throw c10::Error(                                                          \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},                 \
          c10::str(                                                              \
              preamble,                                                          \
              " Error details: ",                                                \
              " Localized_description: ", error.localizedDescription.UTF8String, \
              " Domain: ", error.domain.UTF8String,                              \
              " Code: ", error.code,                                             \
              " User Info: ", error.userInfo.description.UTF8String));           \
    }                                                                            \
  } while (false)

namespace torch {
namespace jit {
namespace mobile {
namespace coreml {

using c10::impl::GenericDict;
using c10::impl::GenericList;
using c10::IValue;

struct CoreMLConfig {
  std::string backend = "CPU";
  bool allow_low_precision = true;
};

bool type_validity(const std::vector<TensorSpec>& specs) {
  for (const TensorSpec& spec : specs) {
    if (spec.dtype != c10::ScalarType::Float) {
      return false;
    }
  }
  return true;
}

void from_json(const nlohmann::json& j, TensorSpec& spec) {
  j[0].get_to(spec.name);
  std::string type_string;
  j[1].get_to(type_string);
  spec.dtype = scalar_type(type_string);
}

void from_json(const nlohmann::json& j, CoreMLConfig& config) {
  j.at("backend").get_to(config.backend);
  std::string allow_low_precision_string;
  j.at("allow_low_precision").get_to(allow_low_precision_string);
  if (allow_low_precision_string == "True") {
    config.allow_low_precision = true;
  } else {
    config.allow_low_precision = false;
  }
}

GenericList pack_outputs(const std::vector<TensorSpec>& output_specs, id<MLFeatureProvider> outputProvider) {
  c10::List<torch::Tensor> outputs;
  for (const TensorSpec& spec : output_specs) {
    NSString *name = [NSString stringWithUTF8String:spec.name.c_str()];
    MLFeatureValue *val = [outputProvider featureValueForName:name];
    std::vector<int64_t> output_shape;
    for (int i = 0; i < val.multiArrayValue.shape.count; ++i) {
      output_shape.emplace_back(val.multiArrayValue.shape[i].integerValue);
    }
    auto tensor = at::empty(IntArrayRef(output_shape), spec.dtype);
    int64_t count = val.multiArrayValue.count;
    memcpy(
      tensor.data_ptr<float>(),
      (float*)val.multiArrayValue.dataPointer,
      count * sizeof(float));
    outputs.push_back(tensor);
  }
  return c10::impl::toList(outputs);
}

class CoreMLBackend: public torch::jit::PyTorchBackendInterface {

 public:
  GenericDict compile(IValue processed, GenericDict method_compile_spec) override {
    const c10::Dict<IValue, IValue> model_dict = processed.toGenericDict();
    const std::string& extra = model_dict.at("extra").toStringRef();
    const std::string& model = model_dict.at("model").toStringRef();
    const std::string& modelID = model_dict.at("hash").toStringRef();

    CoreMLConfig config;
    std::vector<TensorSpec> input_specs;
    std::vector<TensorSpec> output_specs;

    try {
      nlohmann::json extra_json = nlohmann::json::parse(extra);
      config = extra_json["config"].get<CoreMLConfig>();
      input_specs = extra_json["inputs"].get<std::vector<TensorSpec>>();
      output_specs = extra_json["outputs"].get<std::vector<TensorSpec>>();
    } catch (std::exception& exn) {
      TORCH_CHECK(false, "Parsing model dict failed!");
    }

    if (!type_validity(input_specs) || !type_validity(output_specs)) {
      TORCH_CHECK(false, "Compiling model failed, only float type tensors supported");
    }

    if (![PTMCoreMLCompiler compileModel:model modelID:modelID]) {
      TORCH_CHECK(false, "Compiling MLModel failed");
    }

    MLModel *cpuModel = [PTMCoreMLCompiler loadModel:modelID backend:"cpu" allowLowPrecision:NO];

    if (!cpuModel) {
      TORCH_CHECK(false, "Loading MLModel failed");
    }

    NSMutableArray *orderedFeatures = [NSMutableArray array];
    for (TensorSpec& spec : input_specs) {
      NSString *name = [NSString stringWithUTF8String:spec.name.c_str()];
      [orderedFeatures addObject:name];
    }

    PTMCoreMLExecutor *executor = [[PTMCoreMLExecutor alloc] initWithFeatureNames:orderedFeatures];
    executor.model = cpuModel;
    [executor autorelease];

    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
      MLModel *configuredModel = [PTMCoreMLCompiler loadModel:modelID backend:config.backend allowLowPrecision:config.allow_low_precision];
      executor.model = configuredModel ?: cpuModel;
    });

    MLModelWrapper model_wrapper = MLModelWrapper(executor);
    model_wrapper.outputs = output_specs;

    auto model_wrapper_ptr = c10::make_intrusive<MLModelWrapper>(model_wrapper);
    auto handle = IValue::make_capsule(model_wrapper_ptr);

    c10::Dict<IValue, IValue> ret(StringType::get(), c10::AnyType::get());
    ret.insert("forward", handle);
    return c10::impl::toGenericDict(ret);
  }

  GenericList execute(IValue handle, GenericList inputs) override {
    const auto model_wrapper = c10::static_intrusive_pointer_cast<MLModelWrapper>(handle.toCapsule());

    PTMCoreMLExecutor *executor = model_wrapper->executor;
    [executor setInputs:inputs];

    NSError *error;
    id<MLFeatureProvider> outputsProvider = [executor forward:&error];
    if (!outputsProvider) {
      COREML_THROW_IF_ERROR(error, "Error running CoreML inference");
    }

    return pack_outputs(model_wrapper->outputs, outputsProvider);
  }

  bool is_available() override {
#if TARGET_OS_IPHONE
    return [UIDevice currentDevice].systemVersion.floatValue >= 12.0;
#elif TARGET_OS_MAC
    NSOperatingSystemVersion supportedVer = {10, 13, 0};
    return [[NSProcessInfo processInfo] isOperatingSystemAtLeastVersion:supportedVer];
#endif
    return false;
  }
};

static auto cls = torch::jit::backend<CoreMLBackend>("coreml");

struct PTMCoreMLContext : public ContextInterface {
  void setModelCacheDirectory(std::string dir) override {
    [PTMCoreMLCompiler setCacheDirectory:dir];
  }
};

static BackendRegistrar g_coreml_backend(new PTMCoreMLContext());

} // namespace
}
}
}

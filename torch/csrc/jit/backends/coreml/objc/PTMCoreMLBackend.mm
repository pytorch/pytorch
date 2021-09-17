#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/coreml/objc/PTMCoreMLExecutor.h>
#include <torch/script.h>

#import <CoreML/CoreML.h>

namespace torch {
namespace jit {
namespace mobile {
namespace coreml {

enum TensorType {
  Float,
  Double,
  Int,
  Long,
  Undefined,
};

#define DEFINE_SCALAR_TYPES(_) \
  _(Float)                     \
  _(Double)                    \
  _(Int)                       \
  _(Long)                      \
  _(Undefined)

static inline c10::ScalarType scalarType(TensorType type) {
  switch (type) {
#define DEFINE_CASE(x) \
  case x:              \
    return c10::ScalarType::x;
    DEFINE_SCALAR_TYPES(DEFINE_CASE)
#undef DEFINE_CASE
  }
  return c10::ScalarType::Undefined;
}

static id parse(NSString* jsonStr) {
  NSData* data = [jsonStr dataUsingEncoding:NSUTF8StringEncoding];
  NSError* err;
  return [NSJSONSerialization JSONObjectWithData:data options:0 error:&err];
}

struct TensorSpec {
 public:
  TensorSpec() = default;
  TensorSpec(NSArray<NSString*>* spec) {
    TORCH_CHECK(spec.count == 3);
    name_ = spec[0];
    dtype_ = (TensorType)spec[1].intValue;
    NSArray* sizes = parse(spec[2]);
    for (NSString* dim in sizes) {
      sizes_.push_back(dim.integerValue);
    }
  }
  int64_t numel() const {
    return std::accumulate(
        begin(sizes_), end(sizes_), 1, std::multiplies<int64_t>());
  }
  NSString* name() {
    return name_;
  }
  std::vector<int64_t> sizes() {
    return sizes_;
  }
  TensorType dtype() {
    return dtype_;
  }

 private:
  NSString* name_ = @"";
  TensorType dtype_ = TensorType::Float;
  std::vector<int64_t> sizes_{};
};

struct CoreMLConfig {
 public:
  CoreMLConfig() = default;
  CoreMLConfig(NSDictionary* dict) {
    int specVer = [dict[@"spec_ver"] intValue];
    coreMLVersion_ = specVer - 1;
    TORCH_CHECK(
        coreMLVersion_ >= 2, "Only Core ML version 2 and above are supported");
    backend_ = [dict[@"backend"] lowercaseString];
    allow_low_precision_ = [dict[@"allow_low_precision"] boolValue];
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
  int64_t coreMLVersion_ = 2;
  NSString* backend_ = @"cpu";
  bool allow_low_precision_ = true;
};

struct MetaData {
  MetaData() = default;
  MetaData(NSDictionary* dict) {
    torchVer_ = dict[@"torch_ver"];
    coremltoolVer_ = dict[@"coremltool_ver"];
  }
  NSString* torchVer_ = nil;
  NSString* coremltoolVer_ = nil;
};

class API_AVAILABLE(ios(11.0)) CoreMLBackend
    : public torch::jit::PyTorchBackendInterface {
 public:
  CoreMLBackend() : executor_([PTMCoreMLExecutor new]) {}
  c10::impl::GenericDict compile(
      c10::IValue processed,
      c10::impl::GenericDict method_compile_spec) override {
    auto modelDict = processed.toGenericDict();
    const std::string& model = modelDict.at("model").toStringRef();
    const std::string& sha256 = modelDict.at("hash").toStringRef();
    bool result = [executor_ compileMLModel:model identifier:sha256];
    TORCH_CHECK(result, "[CoreML] compiled model failed!");
    NSString* specs = [[NSString alloc]
        initWithCString:modelDict.at("extra").toStringRef().c_str()
               encoding:NSUTF8StringEncoding];
    NSDictionary* dict = parse(specs);
    NSArray* inputs = dict[@"inputs"];
    NSArray* outputs = dict[@"outputs"];
    for (NSArray* input in inputs) {
      inputs_.push_back(TensorSpec(input));
    }
    for (NSArray* output in outputs) {
      outputs_.push_back(TensorSpec(output));
    }
    metaData_ = MetaData(dict[@"metadata"]);
    config_ = CoreMLConfig(dict[@"config"]);

    executor_.allowLowPrecision = config_.allowLowPrecision();
    executor_.backend = config_.backend();
    executor_.coreMLVersion = config_.coreMLVersion();

    return method_compile_spec;
  }

  c10::impl::GenericList execute(
      c10::IValue handle,
      c10::impl::GenericList inputs) override {
    std::vector<PTMCoreMLFeatureSpecs> inputSpecs;
    std::vector<PTMCoreMLFeatureSpecs> outputSpecs;
    int inputSpecIndex = 0;
    // pack the inputs
    for (int i = 0; i < inputs.size(); ++i) {
      auto val = inputs.get(i);
      if (val.isTuple()) {
        auto tuples = val.toTuple()->elements();
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
      auto tensor = at::empty(spec.sizes(), scalarType(spec.dtype()));
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
    if (@available(iOS 11, *)) {
      return true;
    } else {
      return false;
    }
#else
    return false;
#endif
  }

 private:
  std::vector<TensorSpec> inputs_{};
  std::vector<TensorSpec> outputs_{};
  MetaData metaData_;
  CoreMLConfig config_;
  PTMCoreMLExecutor* executor_ = nil;
};

API_AVAILABLE(ios(11.0))
static auto cls = torch::jit::backend<CoreMLBackend>("coreml");

} // namespace
}
}
}

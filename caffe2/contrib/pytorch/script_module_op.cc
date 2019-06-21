#include <caffe2/core/context.h>
#include <caffe2/core/operator.h>
#include <caffe2/utils/math.h>
#include <torch/script.h>
#include "caffe2/core/blob_serialization.h"

namespace caffe2 {

using torch::jit::IValue;
using torch::jit::script::Method;
using torch::jit::script::Module;

namespace {
class ScriptModuleSerializer : public BlobSerializerBase {
 public:
  void Serialize(
      const void* pointer,
      TypeMeta typeMeta,
      const string& name,
      SerializationAcceptor acceptor) override {
    CAFFE_ENFORCE(typeMeta.Match<Module>());

    std::stringstream ss;
    (*static_cast<const Module*>(pointer)).save(ss);

    // NB: wrapping the entire zip archive as one string is probably not a
    // good idea and might be slow. This is meant as a workaround, any proper
    // usage would require splitting out tensor data separately.
    //
    // In the future we can do it by introducing a different "type" string for
    // the more efficient serialization version (if we ever get to that point)
    BlobProto blob_proto;
    blob_proto.set_name(name);
    blob_proto.set_type("torch::jit::script::Module");
    blob_proto.set_content(ss.str());
    acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
  }
};

class ScriptModuleDeserializer : public BlobDeserializerBase {
 public:
  void Deserialize(const BlobProto& proto, Blob* blob) override {
    const auto& serialized = proto.content();
    // TODO: use adapter instead of istream?
    std::stringstream ss;
    ss << serialized;
    ss.seekg(0);
    *blob->GetMutable<Module>() = torch::jit::load(ss);
  }
};

class ScriptModuleLoadOp final : public Operator<CPUContext> {
 public:
  ScriptModuleLoadOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {
    CAFFE_ENFORCE(HasArgument("serialized_binary"));
  }

  bool RunOnDevice() override {
    auto moduleBinary = GetSingleArgument<string>("serialized_binary", "");
    // TODO: use adapter instead of istream?
    std::stringstream ss;
    ss << moduleBinary;
    ss.seekg(0);
    *OperatorBase::Output<Module>(0) = torch::jit::load(ss);
    return true;
  }
};

template <class Context>
class ScriptModuleOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ScriptModuleOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        method_name_(this->template GetSingleArgument<std::string>(
            "method",
            "forward")) {
    // TODO: we could also parse extra arguments here and allow to pass in
    // scalars to the method invocation. But there's probably less blocking need
    // for that.
  }

  static caffe2::Tensor castIValueToTensor(IValue v) {
    return caffe2::Tensor(torch::autograd::Variable(std::move(v).toTensor()).tensor_data());
  }

  bool RunOnDevice() override {
    const auto& module = OperatorBase::Input<Module>(0);
    Method method = module.get_method(method_name_);
    // Assume all inputs are tensor for now
    std::vector<IValue> inputs;
    const int num_inputs = InputSize();
    inputs.reserve(num_inputs);
    for (int i = 1; i < num_inputs; ++i) {
      // jit::Interpreter takes only autograd variables (which have
      // require_grad=False in this case)
      inputs.emplace_back(torch::autograd::make_variable(at::Tensor(Input(i))));
    }
    // We just convert specified inputs. If some of the inputs were omitted and
    // don't have default values, method::operator() is going to complain.
    IValue output = method(inputs);
    if (output.isTuple()) {
      auto elems = std::move(output).toTuple();
      CAFFE_ENFORCE_EQ(elems->elements().size(), OutputSize());
      for (int i = 0; i < elems->elements().size(); ++i) {
        this->SetOutputTensor(i, castIValueToTensor(elems->elements()[i]));
      }
    } else if (output.isTensor()) {
      CAFFE_ENFORCE_EQ(1, OutputSize());
      this->SetOutputTensor(0, castIValueToTensor(std::move(output)));
    } else {
      CAFFE_THROW("Unexpected return type: ", output.tagKind());
    }
    return true;
  }

 private:
  std::string method_name_;
};
} // namespace

CAFFE_KNOWN_TYPE(Module);

REGISTER_BLOB_SERIALIZER(
    (TypeMeta::Id<Module>()),
    ScriptModuleSerializer);
// NB: the first argument to REGISTER_BLOB_DESERIALIZER macro doesn't really
// need to be a real type, it just get converted to string
REGISTER_BLOB_DESERIALIZER(
    torch::jit::script::Module,
    ScriptModuleDeserializer);

OPERATOR_SCHEMA(ScriptModule)
    .NumInputs(1, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .Input(0, "script_module_instance", "Instance of shared_ptr<Module>");
REGISTER_CPU_OPERATOR(ScriptModule, ScriptModuleOp<CPUContext>);
SHOULD_NOT_DO_GRADIENT(ScriptModule);

OPERATOR_SCHEMA(ScriptModuleLoad)
    .NumInputs(0)
    .NumOutputs(1)
    .DisallowInputFillers()
    .Output(0, "script_module_instance", "New instance of shared_ptr<Module>")
    .Arg(
        "serialized_binary",
        "Binary string representing contents of .pt file (zip container)");
REGISTER_CPU_OPERATOR(ScriptModuleLoad, ScriptModuleLoadOp);
NO_GRADIENT(ScriptModuleLoad);

} // namespace caffe2

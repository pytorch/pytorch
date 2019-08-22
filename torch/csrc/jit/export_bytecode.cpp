#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/type_resolver_util.h>

#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/liteinterpreter/frameoutput.h>

#include <torch/csrc/jit/pickle.h>

#include <caffe2/proto/bytecode_pb.h>
#include <caffe2/serialize/inline_container.h>

#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>


namespace torch {
namespace jit {
void convertAndWriteTensor(
    size_t tensor_id,
    const at::Tensor& tensor,
    torch::TensorDef* tensor_proto,
    std::unordered_map<const void*, std::string>& storageMap,
    caffe2::serialize::PyTorchStreamWriter& writer);

namespace mobile {

namespace {
// this is a serializer class which saves bytecode of a frame to a zip file. the
// content of the file is written using PyTorchStreamWriter, for details please
// check caffe2/serialize/inline_container.h. all the records except the last
// one are tensor data, and the last record is a serialized ModelProto, defined
// in caffe2/proto/torch.proto. ModelProto contains all the metadata of the
// model, and it is serialized as json.
class ByteCodeSerializer final {
 public:
  ByteCodeSerializer(const std::string& filename);
  ByteCodeSerializer(std::ostream* ofs);

  void serialize(const script::Module& module);

 private:
  void addConstant(const IValue& val, bytecode::FrameProto& frame_proto);
  std::ofstream ofs_;
  caffe2::serialize::PyTorchStreamWriter writer_;
  std::vector<at::Tensor> tensor_table_;
};

ByteCodeSerializer::ByteCodeSerializer(const std::string& filename)
    : writer_(filename.c_str()) {
  // TODO appropriate support for mmap, right now we still use stream writer
}

ByteCodeSerializer::ByteCodeSerializer(std::ostream* ofs)
    : ofs_(), writer_(ofs) {}

void ByteCodeSerializer::addConstant(const IValue& val, bytecode::FrameProto& frame_proto) {
  auto attribute = frame_proto.add_constants();
  if (val.isInt()) {
    attribute->set_kind(bytecode::ConstantProto::i);
    attribute->set_int_value(val.toInt());
  }
  else if (val.isDouble()) {
    attribute->set_kind(bytecode::ConstantProto::f);
    attribute->set_float_value(val.toDouble());
  }
  else if (val.isTensor()) {
    tensor_table_.push_back(val.toTensor());
    attribute->set_kind(bytecode::ConstantProto::t);
    attribute->set_tensor_id(tensor_table_.size() - 1);
  }
  else if (val.isIntList()) {
    attribute->set_kind(bytecode::ConstantProto::is);
    auto list = val.toIntList();
    for (size_t i = 0; i < list.size(); ++i) {
      attribute->add_int_list(list[i]);
    }
  }
  else if (val.isDoubleList()) {
    attribute->set_kind(bytecode::ConstantProto::fs);
    auto list = val.toDoubleList();
    for (size_t i = 0; i < list.size(); ++i) {
      attribute->add_float_list(list[i]);
    }
  }
  else if (val.isBool()) {
    attribute->set_kind(bytecode::ConstantProto::b);
    attribute->set_bool_value(val.toBool());
  }
  else if (val.isBoolList()) {
    attribute->set_kind(bytecode::ConstantProto::bs);
    auto list = val.toBoolList();
    for (size_t i = 0; i < list.size(); ++i) {
      attribute->add_bool_list(list[i]);
    }
  }
  else if (val.isNone()) {
    attribute->set_kind(bytecode::ConstantProto::n);
  }
  else {
    throw std::runtime_error("Value type of Constant is not supported yet.");
  }
}

void ByteCodeSerializer::serialize(const script::Module& module) {

  auto data = pickle(module.module_object(), &tensor_table_);
  writer_.writeRecord("data.pkl", data.data(), data.size());

  auto compUnit = module.class_compilation_unit();
  auto funcList = compUnit->get_functions();
  for (auto func : funcList) {
    torch::jit::Code code(func->graph());
    auto frame = code.getFrame();
    if (frame == nullptr) continue;
    frame->name = func->name();
    bytecode::FrameProto frame_proto;
    frame_proto.set_name(frame->name);
    frame_proto.set_pc(frame->pc);

    // constants (non-tensor)
    for (const auto& val : frame->constants) {
      addConstant(val, frame_proto);
    }

    // instructions
    for (const auto& ins : frame->instructions) {
      auto ins_proto = frame_proto.add_instructions();
      std::stringstream ss;
      ss << ins.op;
      ins_proto->set_opcode(ss.str());
      ins_proto->set_n(ins.N);
      ins_proto->set_x(ins.X);
    }

    // operators
    for (size_t i = 0; i < frame->operators.size(); ++i) {
      auto op_proto = frame_proto.add_operators();
      auto name = frame->opnames[i].name;
      op_proto->set_name(name);
      op_proto->set_overload_name(frame->opnames[i].overload_name);
    }

    // tensors
    std::unordered_map<const void*, std::string> storageMap;
    size_t tensor_id = 0;
    for (const at::Tensor& t : tensor_table_) {
      auto* tensor_proto = frame_proto.add_tensors();
      convertAndWriteTensor(tensor_id++, t, tensor_proto, storageMap, writer_);
    }

    std::string output;
    // NB: cannot use MessageToJsonString, since fbcode's protobuf is too old
    // be consistent with MessageToJsonString
    std::string url_prefix = "type.googleapis.com";
    std::unique_ptr<::google::protobuf::util::TypeResolver> resolver(
        ::google::protobuf::util::NewTypeResolverForDescriptorPool(
            url_prefix, frame_proto.GetDescriptor()->file()->pool()));
    ::google::protobuf::util::Status convert_result =
        ::google::protobuf::util::BinaryToJsonString(
            resolver.get(),
            url_prefix + "/" + frame_proto.GetDescriptor()->full_name(),
            frame_proto.SerializeAsString(),
            &output);
    if (!convert_result.ok()) {
      std::stringstream ss;
        ss << convert_result;
        AT_ERROR(ss.str());
    }
    std::cout << output << std::endl;
    std::string recordName = frame->name + "/bytecode.json";
    writer_.writeRecord(recordName, output.data(), output.size());
  }

  writer_.writeEndOfFile();
}
} //namespace

void SaveBytecode(
    const script::Module& module,
    const std::string& filename) {
  ByteCodeSerializer serializer(filename);
  serializer.serialize(module);
}

bool TestFunc() {
  std::cout << "test";
  return true;
}

}
}
}

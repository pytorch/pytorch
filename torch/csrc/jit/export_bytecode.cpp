#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/type_resolver_util.h>

#include <torch/csrc/autograd/symbolic.h>
#include <torch/csrc/jit/export.h>
#include <torch/csrc/onnx/onnx.h>

#include <ATen/core/functional.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/import_export_helpers.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/python_print.h>
#include <torch/csrc/jit/pickle.h>
#include <torch/csrc/jit/source_range_serialization.h>

#include <caffe2/core/types.h>
#include <caffe2/proto/caffe2_pb.h>
#include <caffe2/proto/torch_pb.h>
#include <caffe2/proto/bytecode_pb.h>
#include <caffe2/serialize/inline_container.h>
#include <onnx/onnx_pb.h>

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

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
  void addConstant(const IValue& val, bytecode::FrameProto& frame_proto,
                   size_t tensor_id, std::unordered_map<const void*, std::string>& storageMap);
  std::ofstream ofs_;
  caffe2::serialize::PyTorchStreamWriter writer_;
};

ByteCodeSerializer::ByteCodeSerializer(const std::string& filename)
    : writer_(filename.c_str()) {
  // TODO appropriate support for mmap, right now we still use stream writer
}

ByteCodeSerializer::ByteCodeSerializer(std::ostream* ofs)
    : ofs_(), writer_(ofs) {}

void ByteCodeSerializer::addConstant(const IValue& val, bytecode::FrameProto& frame_proto,
                                     size_t tensor_id, std::unordered_map<const void*, std::string>& storageMap) {
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
    attribute->set_kind(bytecode::ConstantProto::t);
    attribute->set_tensor_id(tensor_id);
    auto tensor_proto = frame_proto.add_tensors();
    convertAndWriteTensor(tensor_id++, val.toTensor(), tensor_proto, storageMap, writer_);
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
  auto compUnit = module.class_compilation_unit();
  auto funcList = compUnit->get_functions();
  for (auto func : funcList) {
    auto funcName = func->name();
    torch::jit::Code code(func->graph());
  }

//  auto method = module.get_method("forward");

//  auto frame = method.function().get_executor().getFrame();

//  if (frame == nullptr) return;

//  bytecode::FrameProto frame_proto;
//  frame_proto.set_name(frame->name);
//  frame_proto.set_pc(frame->pc);

//  // constants
//  std::unordered_map<const void*, std::string> storageMap;
//  size_t tensor_id = 0;
//  for (const auto& val : frame->constants) {
//    addConstant(val, frame_proto, tensor_id, storageMap);
//  }

//  // instructions. special treatment on attributes
//  for (const auto& ins : frame->instructions) {
//    auto ins_proto = frame_proto.add_instructions();
//    std::stringstream ss;
//    ss << ins.op;
//    ins_proto->set_opcode(ss.str());
//    ins_proto->set_n(ins.N);
//    ins_proto->set_x(ins.X);
//  }

//  // operators and parameters
//  Stack outstack;
//  for (size_t i = 0; i < frame->operators.size(); ++i) {
//    auto op_proto = frame_proto.add_operators();
//    auto name = frame->opnames[i].name;
//  //    if (name == "prim::GetAttr") {
//  //      if (outstack.empty()) {
//  //        outstack.emplace_back(module.module_object());
//  //      }
//  //      frame->operators[i](outstack);
//  //      auto val = outstack.back();
//    //      if (val.isObject()) {
//    //        std::cout << val << std::endl;
//    //      }
//    //      else {
//    //        // Change it to LOADC
//    //        op_proto->set_name();
//    //      }
//    //    }
//    op_proto->set_name(name);
//    op_proto->set_overload_name(frame->opnames[i].overload_name);
//  }

//  std::string output;
//  // NB: cannot use MessageToJsonString, since fbcode's protobuf is too old
//  // be consistent with MessageToJsonString
//  std::string url_prefix = "type.googleapis.com";
//  std::unique_ptr<::google::protobuf::util::TypeResolver> resolver(
//      ::google::protobuf::util::NewTypeResolverForDescriptorPool(
//          url_prefix, frame_proto.GetDescriptor()->file()->pool()));
//  ::google::protobuf::util::Status convert_result =
//      ::google::protobuf::util::BinaryToJsonString(
//          resolver.get(),
//          url_prefix + "/" + frame_proto.GetDescriptor()->full_name(),
//          frame_proto.SerializeAsString(),
//          &output);
//  if (!convert_result.ok()) {
//    std::stringstream ss;
//    ss << convert_result;
//    AT_ERROR(ss.str());
//  }
//  std::cout << output << std::endl;
//  writer_.writeRecord("bytecode.json", output.data(), output.size());
//  writer_.writeEndOfFile();
}

void SaveBytecode(
    const script::Module& module,
    const std::string& filename) {
//#ifdef FBCODE_CAFFE2
//  ScriptModuleSerializer serializer(&out);
//#else
//  ScriptModuleSerializer2 serializer(&out);
//#endif

//  serializer.serialize(module, extra_files);
}
}
}
}

#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/type_resolver_util.h>

#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/liteinterpreter/frameoutput.h>

#include <torch/csrc/jit/pickle.h>

#include <caffe2/serialize/inline_container.h>

#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>


namespace torch {
namespace jit {
void writeArchive(const std::string& archive_name, const IValue& value,
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
  std::ofstream ofs_;
  caffe2::serialize::PyTorchStreamWriter writer_;
};

ByteCodeSerializer::ByteCodeSerializer(const std::string& filename)
    : writer_(filename.c_str()) {
  // TODO appropriate support for mmap, right now we still use stream writer
}

ByteCodeSerializer::ByteCodeSerializer(std::ostream* ofs)
    : ofs_(), writer_(ofs) {}

void ByteCodeSerializer::serialize(const script::Module& module) {
  writeArchive("data", module.module_object(), writer_);

  auto compUnit = module.class_compilation_unit();
  auto funcList = compUnit->get_functions();

  for (auto func : funcList) {
    torch::jit::Code code(func->graph());
    auto frame = code.getFrame();
    if (frame == nullptr) continue;
    frame->name = func->name();

    auto constants = c10::ivalue::Tuple::create(std::move(frame->constants));
    writeArchive(frame->name + "/constants", constants, writer_);

    // instructions
    std::vector<IValue> inss;
    for (const auto& ins : frame->instructions) {
      std::stringstream ss;
      ss << ins.op;
      std::vector<IValue> insv{ss.str(), ins.N, ins.X};
      inss.emplace_back(c10::ivalue::Tuple::create(std::move(insv)));
    }
    auto instructions = c10::ivalue::Tuple::create(std::move(inss));

    // operators
    std::vector<IValue> opss;
    for (const auto& opname : frame->opnames) {
      opss.emplace_back(c10::ivalue::Tuple::create({opname.name, opname.overload_name}));
    }
    auto operators = c10::ivalue::Tuple::create(std::move(opss));

    auto elements = c10::ivalue::Tuple::create({instructions, operators});
    std::vector<at::Tensor> temp_table;
    auto bdata = pickle(elements, &temp_table);
    writer_.writeRecord(frame->name + "/bytecode.pkl", bdata.data(), bdata.size());
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

}
}
}

#include "import.h"
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/script/compilation_unit.h>
#include <torch/csrc/jit/unpickler.h>
#include <caffe2/serialize/inline_container.h>


#include <fstream>
#include <string>
#include <vector>

// The import process to serialize the bytecode package.
// An example for bytecode.pkl of a small mobile_module looks like:
//  (('__torch__.m.add_it',
//    (('instructions',
//      (('STOREN', 1, 2),
//       ('MOVE', 1, 0),
//       ('GET_ATTR', 0, 0),
//       ('MOVE', 2, 0),
//       ('LOADC', 0, 0),
//       ('OP', 0, 0),
//       ('LOADC', 1, 0),
//       ('LOADC', 0, 0),
//       ('OP', 1, 0),
//       ('RET', 0, 0))),
//     ('operators', (('_aten::add', 'Tensor'), ('_aten::add', 'Scalar'))),
//     ('constants', (1, 4)),
//     ('register_size', 2))),)

// Note that currently the backward compatibility is not supported by bytecode.
// This format and process need to be revisted and redesigned if we want to
// suppot backward compatibility in future.

namespace torch {
namespace jit {
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::ReadAdapterInterface;

OpCode parseOpCode(const char *str);
namespace {
void parseMethods(const std::vector<IValue>& vals, std::shared_ptr<mobile::CompilationUnit> mcu) {
  for (const auto& element : vals) {
    const auto& m_tuple = element.toTuple()->elements();

    auto function = std::unique_ptr<mobile::Function>(new mobile::Function(
        c10::QualifiedName(m_tuple[0].toString()->string())));
    auto comps = m_tuple[1].toTuple()->elements();

    // The sequence of the named tuple is 0: instructions, 1: operators,
    // 2: constants, 3: register_size
    auto named_ins = comps[0].toTuple()->elements();
    auto ins_name = named_ins[0].toString()->string();
    TORCH_CHECK(ins_name == "instructions",
                "instruction is expected, but get", ins_name);
    auto ins_list = named_ins[1].toTuple()->elements();
    for (const auto& ins : ins_list) {
      auto ins_item = ins.toTuple()->elements();
      TORCH_CHECK(ins_item.size() == 3,
                  "There should be three parts in an instruction.");
      OpCode op_code = parseOpCode(ins_item[0].toString()->string().c_str());
      function->append_instruction(op_code, ins_item[1].toInt(),
                                ins_item[2].toInt());
    }

    auto named_ops = comps[1].toTuple()->elements();
    auto ops_name = named_ops[0].toString()->string();
    TORCH_CHECK(ops_name == "operators",
                "operator is expected, but get", ops_name);
    auto ops_list = named_ops[1].toTuple()->elements();
    for (const auto& op : ops_list) {
      auto op_item = op.toTuple()->elements();
      TORCH_CHECK(op_item.size() == 2,
                  "There should be two parts in an operator name.");
      function->append_operator(op_item[0].toString()->string(),
                           op_item[1].toString()->string());
    }

    auto named_consts = comps[2].toTuple()->elements();
    auto consts_name = named_consts[0].toString()->string();
    TORCH_CHECK(consts_name == "constants",
                "constant is expected, but get", consts_name);
    auto consts_list = named_consts[1].toTuple()->elements();
    for (const auto& constant : consts_list) {
      function->append_constant(constant);
    }

    auto named_agg_size = comps[3].toTuple()->elements();
    auto size_name = named_agg_size[0].toString()->string();
    TORCH_CHECK(size_name == "register_size",
                "register_size is expected, but get", ops_name);
    function->set_register_size(named_agg_size[1].toInt());

    mcu->register_function(std::move(function));
  }
}

// The deserializer class which loads the bytecode package from bc files.
class BytecodeDeserializer final {
 public:
  explicit BytecodeDeserializer(std::unique_ptr<PyTorchStreamReader> reader);
  mobile::Module deserialize(c10::optional<at::Device> device);

 private:
  c10::IValue readArchive(const std::string& archive_name);
  std::shared_ptr<script::CompilationUnit> compilation_unit_;
  std::unordered_set<std::string> imported_libs_;
  std::unique_ptr<PyTorchStreamReader> reader_;
  c10::optional<at::Device> device_;
};

BytecodeDeserializer::BytecodeDeserializer(std::unique_ptr<PyTorchStreamReader> reader)
    : compilation_unit_(std::make_shared<script::CompilationUnit>()), reader_(std::move(reader)) {}

mobile::Module BytecodeDeserializer::deserialize(c10::optional<at::Device> device) {
  device_ = device;
  auto bvals = readArchive("bytecode").toTuple()->elements();
  auto mcu = std::make_shared<mobile::CompilationUnit>();
  parseMethods(bvals, mcu);

  return mobile::Module(readArchive("data").toObject(), mcu);
}

c10::IValue BytecodeDeserializer::readArchive(const std::string& archive_name) {
  std::stringstream picklename;
  picklename << archive_name << ".pkl";
  at::DataPtr pickle_ptr;
  size_t pickle_size;
  std::tie(pickle_ptr, pickle_size) = reader_->getRecord(picklename.str());

  size_t bytes_read = 0;
  auto data = reinterpret_cast<const char*>(pickle_ptr.get());
  auto reader = [&](char* buffer, size_t len) {
    if (bytes_read + len > pickle_size) {
      return false;
    }
    // Copy len bytes into buffer
    const char* start = data + bytes_read;
    std::memcpy(buffer, start, len);
    bytes_read += len;
    return true;
  };

  auto class_resolver = [&](const c10::QualifiedName& qn) {
    if (compilation_unit_->get_class(qn) == nullptr) {
      auto typeptr = ClassType::create(qn, compilation_unit_, true);
      compilation_unit_->register_type(typeptr);
    }
    return c10::StrongTypePtr(
        compilation_unit_, compilation_unit_->get_class(qn));
  };

  auto obj_loader = [&](at::StrongTypePtr type, IValue input) {
    auto dict = std::move(input).toGenericDict();
    size_t ndict = dict.size();
    auto obj = c10::ivalue::Object::create(type, ndict);
    auto it = dict.begin();
    for (size_t i = 0; i < ndict; ++i) {
      obj->setSlot(i, it->value());
      ++it;
    }
    return obj;
  };

  auto read_record = [&](const std::string& name) {
    std::stringstream ss;
    ss << archive_name << "/" << name;
    return std::get<0>(reader_->getRecord(ss.str()));
  };

  Unpickler unpickler(reader, std::move(class_resolver),
                      std::move(obj_loader), std::move(read_record), device_);
  return unpickler.parse_ivalue();
}

} // namespace

mobile::Module _load_for_mobile(
    std::istream& in,
    c10::optional<at::Device> device) {
  std::unique_ptr<IStreamAdapter> rai =
      caffe2::make_unique<IStreamAdapter>(&in);
  auto module = _load_for_mobile(std::move(rai), device);
  return module;
}

mobile::Module _load_for_mobile(
    const std::string& filename,
    c10::optional<at::Device> device) {
  std::unique_ptr<FileAdapter> rai = caffe2::make_unique<FileAdapter>(filename);
  auto module = _load_for_mobile(std::move(rai), device);
  return module;
}

mobile::Module _load_for_mobile(
    std::unique_ptr<ReadAdapterInterface> rai,
    c10::optional<c10::Device> device) {
  auto reader = torch::make_unique<PyTorchStreamReader>(std::move(rai));
  BytecodeDeserializer deserializer(std::move(reader));
  return deserializer.deserialize(device);
}

} // namespace jit
} // namespace torch

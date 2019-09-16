#include "import_bytecode.h"
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/script/compilation_unit.h>
#include <torch/csrc/jit/unpickler.h>
#include <caffe2/serialize/inline_container.h>


#include <fstream>
#include <string>
#include <vector>

namespace torch {
namespace jit {
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::ReadAdapterInterface;

OpCode str2OpCode(const char *str);
namespace {
void parseMethods(const std::vector<IValue>& vals, mobile::Bytecode& bc) {
  for (const auto& element : vals) {
    mobile::Method method;
    const auto& m_tuple = element.toTuple()->elements();
    method.set_name(m_tuple[0].toString()->string());
    auto comps = m_tuple[1].toTuple()->elements();

    // The sequence of the named tuple is 0: instructions, 1: operators,
    // 2: constants, 3: agg_output_size
    auto named_ins = comps[0].toTuple()->elements();
    auto ins_name = named_ins[0].toString()->string();
    TORCH_CHECK(ins_name == "instructions",
                "instruction is expected, but get", ins_name);
    auto ins_list = named_ins[1].toTuple()->elements();
    for (const auto& ins : ins_list) {
      auto ins_item = ins.toTuple()->elements();
      TORCH_CHECK(ins_item.size() == 3,
                  "There should be three parts in an instruction.");
      OpCode op_code = str2OpCode(ins_item[0].toString()->string().c_str());
      method.append_instruction(op_code, ins_item[1].toInt(),
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
      method.append_opname(op_item[0].toString()->string(),
                           op_item[1].toString()->string());
    }

    auto named_consts = comps[2].toTuple()->elements();
    auto consts_name = named_consts[0].toString()->string();
    TORCH_CHECK(consts_name == "constants",
                "constant is expected, but get", consts_name);
    auto consts_list = named_consts[1].toTuple()->elements();
    for (const auto& constant : consts_list) {
      method.append_constant(constant);
    }

    auto named_agg_size = comps[3].toTuple()->elements();
    auto size_name = named_agg_size[0].toString()->string();
    TORCH_CHECK(size_name == "agg_output_size",
                "agg_output_size is expected, but get", ops_name);
    method.resize_registers(named_agg_size[1].toInt());

    bc.append_method(method);
  }
}

void parseSlotNames(const std::vector<IValue>& vals,
                    std::unordered_map<std::string, IValue>& name_table) {
  for (const auto& val : vals) {
    auto tval = val.toTuple()->elements();
    TORCH_CHECK(tval.size() == 2,
                "Slot name has qualified name: tuple format.");
    name_table[tval[0].toString()->string()] = tval[1];
  }
}

// The deserializer class which loads the bytecode package from bc files.
class BytecodeDeserializer final {
 public:
  explicit BytecodeDeserializer(std::unique_ptr<PyTorchStreamReader> reader);
  mobile::Bytecode deserialize(c10::optional<at::Device> device);

 private:
  c10::IValue readArchive(const std::string& archive_name);
  std::shared_ptr<script::CompilationUnit> compilation_unit_;
  std::unordered_set<std::string> imported_libs_;
  std::unique_ptr<PyTorchStreamReader> reader_;
  c10::optional<at::Device> device_;
  // Key: qualified name, value: list of slot names in order
  std::unordered_map<std::string, IValue> slot_name_table_;
};

BytecodeDeserializer::BytecodeDeserializer(std::unique_ptr<PyTorchStreamReader> reader)
    : compilation_unit_(std::make_shared<script::CompilationUnit>()), reader_(std::move(reader)) {}

mobile::Bytecode BytecodeDeserializer::deserialize(c10::optional<at::Device> device) {
  device_ = device;
  // vals[0]: methods. vals[1]: slot_names
  auto bvals = readArchive("bytecode").toTuple()->elements();
  TORCH_CHECK(bvals.size() == 2,
              "Bytecode has two components: methods and slot_names.");
  auto mvals = bvals[0].toTuple()->elements();
  TORCH_CHECK(mvals.size() == 2,
              "Methods has \"methods\": tuple format.");
  TORCH_CHECK(mvals[0].toString()->string() == "methods",
              "Field of \"methods\" is expected.");
  mobile::Bytecode bc;
  parseMethods(mvals[1].toTuple()->elements(), bc);

  auto svals = bvals[1].toTuple()->elements();
  TORCH_CHECK(svals.size() == 2,
              "Slot_names has \"slots_names\": tuple format.");
  TORCH_CHECK(svals[0].toString()->string() == "slots_names",
              "Field of \"slot_names\" is expected.");
  parseSlotNames(svals[1].toTuple()->elements(), slot_name_table_);
  bc.set_object(readArchive("data").toObject());
  return bc;
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

  auto obj_callback = [&](const c10::QualifiedName& qn, IValue input) {
    if (compilation_unit_->get_class(qn) == nullptr) {
      auto typeptr = ClassType::create(qn, compilation_unit_, true);
      compilation_unit_->register_type(typeptr);
    }
    at::StrongTypePtr type = c10::StrongTypePtr(
        compilation_unit_, compilation_unit_->get_class(qn));
    auto dict = std::move(input).toGenericDict();
    size_t ndict = dict.size();
    auto it = slot_name_table_.find(qn.qualifiedName());
    TORCH_CHECK(it != slot_name_table_.end(),
                "qn.qualifiedName() not found.");
    auto slot_names = it->second.toTuple()->elements();
    TORCH_CHECK(slot_names.size() == ndict,
                "The size of slot_names does not match dict size.");
    auto obj = c10::ivalue::Object::create(type, ndict);
    for (size_t i = 0; i < ndict; ++i) {
      obj->setSlot(i, dict.at(slot_names[i].toString()->string()));
    }
    return obj;
  };

  auto read_record = [&](const std::string& name) {
    std::stringstream ss;
    ss << archive_name << "/" << name;
    return std::get<0>(reader_->getRecord(ss.str()));
  };

  Unpickler unpickler(
      reader, std::move(obj_callback), std::move(read_record), device_, true);
  return unpickler.parse_ivalue();
}

} // namespace

mobile::Bytecode load_bytecode(
    std::istream& in,
    c10::optional<at::Device> device) {
  std::unique_ptr<IStreamAdapter> rai =
      caffe2::make_unique<IStreamAdapter>(&in);
  auto bc = load_bytecode(std::move(rai), device);
  return bc;
}

mobile::Bytecode load_bytecode(
    const std::string& filename,
    c10::optional<at::Device> device) {
  std::unique_ptr<FileAdapter> rai = caffe2::make_unique<FileAdapter>(filename);
  auto module = load_bytecode(std::move(rai), device);
  return module;
}

mobile::Bytecode load_bytecode(
    std::unique_ptr<ReadAdapterInterface> rai,
    c10::optional<c10::Device> device) {
  auto reader = torch::make_unique<PyTorchStreamReader>(std::move(rai));
  BytecodeDeserializer deserializer(std::move(reader));
  return deserializer.deserialize(device);
}

} // namespace jit
} // namespace torch

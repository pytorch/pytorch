#include <ATen/core/ivalue.h>
#include <caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/api/compilation_unit.h> // removed after using simple type_resolver/obj_loader
#include <torch/csrc/jit/mobile/import.h> // removed after using simple type_resolver/obj_loader
#include <torch/csrc/jit/mobile/model_compatibility.h>
#include <torch/csrc/jit/serialization/import_read.h>

#include <string>
#include <vector>

namespace torch {
namespace jit {

using caffe2::serialize::FileAdapter;
using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::ReadAdapterInterface;

namespace {

c10::IValue readArchive(
    const std::string& archive_name,
    PyTorchStreamReader& stream_reader) {
  c10::optional<at::Device> device;
  std::shared_ptr<CompilationUnit> compilation_unit =
      std::make_shared<CompilationUnit>();
  auto type_resolver = [&](const c10::QualifiedName& qn) {
    return typeResolverGeneral(qn, compilation_unit);
  };

  std::shared_ptr<mobile::CompilationUnit> mobile_compilation_unit =
      std::make_shared<mobile::CompilationUnit>();
  auto obj_loader = [&](at::StrongTypePtr type, IValue input) {
    return objLoaderGeneral(type, input, mobile_compilation_unit);
  };

  auto ivalues = torch::jit::readArchiveAndTensors(
      archive_name, type_resolver, obj_loader, device, stream_reader);
  return ivalues;
}

bool check_zip_file(std::shared_ptr<ReadAdapterInterface> rai) {
  std::array<uint8_t, 2> first_short{};
  static constexpr uint8_t first_slot = 0x80;
  static constexpr uint8_t second_slot = 0x02;
  rai->read(
      /*pos=*/0,
      /*buf=*/&first_short,
      /*n=*/2,
      /*what=*/"checking archive");
  if (first_short[0] == first_slot && first_short[1] == second_slot) {
    // NB: zip files by spec can start with any data, so technically they might
    // start with 0x80 0x02, but in practice zip files start with a file entry
    // which begins with 0x04034b50. Furthermore, PyTorch will never produce zip
    // files that do not start with the file entry, so it is relatively safe to
    // perform this check.
    TORCH_WARN("The zip file might be problematic. Please check it again.");
    return false;
  }
  return true;
}

std::vector<IValue> get_bytecode_values(PyTorchStreamReader& reader) {
  std::vector<IValue> bytecode_values;
  bytecode_values = readArchive("bytecode", reader).toTuple()->elements();
  return bytecode_values;
}

} // namespace

/********************** Bytecode Version **********************/

// Forward declare
int64_t _get_model_bytecode_version(std::vector<IValue> bytecode_ivalues);

int64_t _get_model_bytecode_version(std::istream& in) {
  std::unique_ptr<IStreamAdapter> rai = std::make_unique<IStreamAdapter>(&in);
  return _get_model_bytecode_version(std::move(rai));
}

int64_t _get_model_bytecode_version(const std::string& filename) {
  std::unique_ptr<FileAdapter> rai = std::make_unique<FileAdapter>(filename);
  return _get_model_bytecode_version(std::move(rai));
}

int64_t _get_model_bytecode_version(std::shared_ptr<ReadAdapterInterface> rai) {
  if (!check_zip_file(rai)) {
    return -1;
  }
  PyTorchStreamReader reader(std::move(rai));
  auto bytecode_values = get_bytecode_values(reader);
  return _get_model_bytecode_version(bytecode_values);
}

int64_t _get_model_bytecode_version(std::vector<IValue> bytecode_ivalues) {
  if (!bytecode_ivalues.empty() && bytecode_ivalues[0].isInt()) {
    int64_t model_version = bytecode_ivalues[0].toInt();
    return model_version;
  }
  TORCH_WARN("Fail to get bytecode version.");
  return -1;
}

/********************** Operators and Info **********************/

// Forward declare
std::unordered_map<std::string, OperatorInfo> _get_model_ops_and_info(
    std::vector<IValue> bytecode_ivalues);

std::unordered_map<std::string, OperatorInfo> _get_model_ops_and_info(
    std::istream& in) {
  std::unique_ptr<IStreamAdapter> rai = std::make_unique<IStreamAdapter>(&in);
  return _get_model_ops_and_info(std::move(rai));
}

std::unordered_map<std::string, OperatorInfo> _get_model_ops_and_info(
    const std::string& filename) {
  std::unique_ptr<FileAdapter> rai = std::make_unique<FileAdapter>(filename);
  return _get_model_ops_and_info(std::move(rai));
}

std::unordered_map<std::string, OperatorInfo> _get_model_ops_and_info(
    std::shared_ptr<ReadAdapterInterface> rai) {
  if (!check_zip_file(rai)) {
    TORCH_WARN("Failed to open zip file for model ops.");
    return std::unordered_map<std::string, OperatorInfo>{};
  }
  PyTorchStreamReader reader(std::move(rai));
  auto bytecode_values = get_bytecode_values(reader);
  return _get_model_ops_and_info(bytecode_values);
}

std::unordered_map<std::string, OperatorInfo> _get_model_ops_and_info(
    std::vector<IValue> bytecode_ivalues) {
  if (_get_model_bytecode_version(bytecode_ivalues) < 6) {
    TORCH_WARN(
        "This model likely has an older bytecode version without operator schema information. Please re-export your model to generate it");
  }
  std::unordered_map<std::string, OperatorInfo> result;
  if (!bytecode_ivalues.empty()) {
    // loop over all the functions in the bytecode
    for (int i = 1; i < bytecode_ivalues.size(); i++) {
      // descend to the operators list
      auto method_tuple = bytecode_ivalues[i].toTuple()->elements();
      auto operators_tuple = method_tuple[1].toTuple()->elements()[1];
      auto operators = operators_tuple.toTuple()->elements()[1];
      for (auto& op_tuple : operators.toTuple()->elements()) {
        auto op = op_tuple.toTuple()->elements();

        // grab name
        std::string op_name = op[0].toStringRef();
        std::string op_overload_name = op[1].toStringRef();
        if (op_overload_name != "") {
          op_name.append(".");
          op_name.append(op_overload_name);
        }

        // grab schema
        int num_args = -1;
        if (op.size() > 2) {
          num_args = op[2].toInt();
        }
        result.emplace(op_name, OperatorInfo{num_args});
      }
    }
    return result;
  }
  TORCH_WARN("Failed to get model ops and info.");
  return result;
}

} // namespace jit
} // namespace torch

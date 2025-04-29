#include <ATen/core/ivalue.h>
#include <caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/api/compilation_unit.h> // removed after using simple type_resolver/obj_loader
#include <torch/csrc/jit/mobile/compatibility/model_compatibility.h>
#include <torch/csrc/jit/mobile/file_format.h>
#include <torch/csrc/jit/mobile/flatbuffer_loader.h>
#include <torch/csrc/jit/mobile/import.h> // removed after using simple type_resolver/obj_loader
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/serialization/import_export_constants.h>
#include <torch/csrc/jit/serialization/import_read.h>

#include <caffe2/serialize/in_memory_adapter.h>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace c10 {
TypePtr parseType(const std::string& pythonStr);
} // namespace c10

namespace torch::jit {

using caffe2::serialize::FileAdapter;
using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::ReadAdapterInterface;

c10::IValue readArchive(
    const std::string& archive_name,
    PyTorchStreamReader& stream_reader) {
  std::optional<at::Device> device;
  std::shared_ptr<CompilationUnit> compilation_unit =
      std::make_shared<CompilationUnit>();

  // TODO (T90180710): Simplify type_resolver and obj_loader when getting
  // bytecode version from model
  auto type_resolver = [&](const c10::QualifiedName& qn) {
    return typeResolverMobile(qn, compilation_unit);
  };

  std::shared_ptr<mobile::CompilationUnit> mobile_compilation_unit =
      std::make_shared<mobile::CompilationUnit>();
  auto obj_loader = [&](const at::StrongTypePtr& type, const IValue& input) {
    return objLoaderMobile(type, input, *mobile_compilation_unit);
  };
  bool bytecode_tensor_in_constants_archive =
      (archive_name == "bytecode" && !isTensorInBytecodeArchive(stream_reader));
  auto ivalues = torch::jit::readArchiveAndTensors(
      archive_name,
      /*pickle_prefix=*/"",
      /*tensor_prefix=*/
      bytecode_tensor_in_constants_archive ? "constants/" : "",
      type_resolver,
      obj_loader,
      device,
      stream_reader,
      nullptr);
  return ivalues;
}

std::vector<IValue> get_bytecode_ivalues(PyTorchStreamReader& reader) {
  return std::move(*readArchive("bytecode", reader).toTuple()).elements().vec();
}

/********************** Bytecode **********************/

// Forward declare
uint64_t _get_model_bytecode_version(
    const std::vector<IValue>& bytecode_ivalues);
static uint64_t _get_model_bytecode_version_from_bytes(char* data, size_t size);

uint64_t _get_model_bytecode_version(std::istream& in) {
  auto orig_pos = in.tellg();
  in.seekg(0, in.beg);
  auto [data, size] = get_stream_content(in);
  in.seekg(orig_pos, in.beg);
  return _get_model_bytecode_version_from_bytes(data.get(), size);
}

uint64_t _get_model_bytecode_version(const std::string& filename) {
  std::ifstream ifile(filename);
  return _get_model_bytecode_version(ifile);
}

uint64_t _get_model_bytecode_version(
    const std::shared_ptr<ReadAdapterInterface>& rai) {
  auto [data, size] = get_rai_content(rai.get());
  return _get_model_bytecode_version_from_bytes(data.get(), size);
}

static uint64_t _get_model_bytecode_version_zip(
    std::shared_ptr<ReadAdapterInterface> rai) {
  if (!check_zip_file(rai)) {
    TORCH_CHECK(
        false,
        "Failed to open .ptl file please ensure the model was exported for mobile");
  }
  PyTorchStreamReader reader(std::move(rai));
  auto bytecode_values = get_bytecode_ivalues(reader);
  return _get_model_bytecode_version(bytecode_values);
}

uint64_t _get_model_bytecode_version_from_bytes(char* data, size_t size) {
  TORCH_CHECK(data != nullptr, "Pointer to bytes is null.");
  TORCH_CHECK(size >= kFileFormatHeaderSize, "Unrecognized data format");
  auto format = getFileFormat(data);
  switch (format) {
    case FileFormat::FlatbufferFileFormat: {
      return get_bytecode_version_from_bytes(data);
    }
    case FileFormat::ZipFileFormat: {
      auto rai =
          std::make_unique<caffe2::serialize::MemoryReadAdapter>(data, size);
      auto version = _get_model_bytecode_version_zip(std::move(rai));
      return version;
    }

    default:
      TORCH_CHECK(false, "Unrecognized data format");
  }
}

uint64_t _get_model_bytecode_version(
    const std::vector<IValue>& bytecode_ivalues) {
  if (!bytecode_ivalues.empty() && bytecode_ivalues[0].isInt()) {
    int64_t model_version = bytecode_ivalues[0].toInt();
    TORCH_CHECK(
        model_version > 0,
        "Expected model bytecode version > 0 got ",
        model_version);
    return static_cast<uint64_t>(model_version);
  }
  TORCH_CHECK(false, "Failed to get bytecode version.");
}

/********************** Operator Version **********************/

uint64_t _get_model_operator_version(
    PyTorchStreamReader& reader); // Forward Declare

uint64_t _get_model_operator_version(std::istream& in) {
  std::unique_ptr<IStreamAdapter> rai = std::make_unique<IStreamAdapter>(&in);
  return _get_model_operator_version(std::move(rai));
}

uint64_t _get_model_operator_version(const std::string& filename) {
  std::unique_ptr<FileAdapter> rai = std::make_unique<FileAdapter>(filename);
  return _get_model_operator_version(std::move(rai));
}

uint64_t _get_model_operator_version(
    std::shared_ptr<ReadAdapterInterface> rai) {
  if (!check_zip_file(rai)) {
    TORCH_CHECK(
        false,
        "Failed to open .ptl file please ensure the model was exported for mobile");
  }
  PyTorchStreamReader reader(std::move(rai));
  return _get_model_operator_version(reader);
}

uint64_t _get_model_operator_version(PyTorchStreamReader& reader) {
  return reader.version();
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
  auto bytecode_values = get_bytecode_ivalues(reader);
  return _get_model_ops_and_info(bytecode_values);
}

/* A function to retrieve the root (top level) operators of a model and their
 * corresponding compatibility info. These root operators can call other
 * operators within them (traced ops), and a root op can call many different
 * traced ops depending on internal code paths in the root op. These traced ops
 * are not returned by this function. Those operators are abstracted into the
 * runtime as an implementation detail (and the traced ops themselves can also
 * call other operators) making retrieving them difficult and their value from
 * this api negligible since they will differ between which runtime version the
 * model is run on. Because of this, there is a false positive this api can't
 * prevent in a compatibility usecase. All the root ops of a model are present
 * in a target runtime, but not all the traced ops are which prevents a model
 * from being able to run.
 **/
std::unordered_map<std::string, OperatorInfo> _get_model_ops_and_info(
    std::vector<IValue> bytecode_ivalues) {
  constexpr uint64_t min_version_with_schema = 6;
  if (_get_model_bytecode_version(bytecode_ivalues) < min_version_with_schema) {
    TORCH_WARN(
        "Only models with bytecode version 6 and above contain operator schema information. Please re-export your model to generate it");
  }
  std::unordered_map<std::string, OperatorInfo> result;
  if (bytecode_ivalues.empty()) {
    TORCH_WARN("Failed to get model ops and info.");
    return result;
  }
  // loop over all the functions in the bytecode
  for (const auto i : c10::irange(1, bytecode_ivalues.size())) {
    // descend to the operators list
    const auto& method_tuple = bytecode_ivalues.at(i).toTupleRef().elements();
    auto operators_tuple = method_tuple.at(1).toTupleRef().elements()[1];
    auto operators = operators_tuple.toTupleRef().elements()[1];
    for (auto& op_tuple : operators.toTupleRef().elements()) {
      const auto& op = op_tuple.toTupleRef().elements();

      // grab name
      std::string op_name = op.at(0).toStringRef();
      std::string op_overload_name = op.at(1).toStringRef();
      if (!op_overload_name.empty()) {
        op_name.append(".");
        op_name.append(op_overload_name);
      }

      // grab schema size
      if (op.size() > 2) {
        result.emplace(op_name, OperatorInfo{(int)op.at(2).toInt()});
      } else { // no schema information use default
        result.emplace(op_name, OperatorInfo{});
      }
    }
  }
  return result;
}

/********************** Get Type Table **********************/

// Forward declare
std::unordered_set<std::string> _get_mobile_model_contained_types(
    const std::vector<IValue>& bytecode_ivalues);

std::unordered_set<std::string> _get_mobile_model_contained_types(
    std::istream& in) {
  std::unique_ptr<IStreamAdapter> rai = std::make_unique<IStreamAdapter>(&in);
  return _get_mobile_model_contained_types(std::move(rai));
}

std::unordered_set<std::string> _get_mobile_model_contained_types(
    const std::string& filename) {
  std::unique_ptr<FileAdapter> rai = std::make_unique<FileAdapter>(filename);
  return _get_mobile_model_contained_types(std::move(rai));
}

std::unordered_set<std::string> _get_mobile_model_contained_types(
    std::shared_ptr<ReadAdapterInterface> rai) {
  if (!check_zip_file(rai)) {
    TORCH_CHECK(
        false,
        "Failed to open .ptl file please ensure the model was exported for mobile");
  }
  PyTorchStreamReader reader(std::move(rai));
  auto bytecode_values = get_bytecode_ivalues(reader);
  return _get_mobile_model_contained_types(bytecode_values);
}

// Get deduplicate type table given bytecode, and each string is a atomic type,
// like str, Tensor and etc. For example,
// input: "Dict[int, Tuple[Tensor, Tensor, Tensor]]"
// output: {Dict, int, Tuple, Tensor}
std::unordered_set<std::string> _get_mobile_model_contained_types(
    const std::vector<IValue>& bytecode_ivalues) {
  std::unordered_set<std::string> contained_types;
  for (const auto i : c10::irange(1, bytecode_ivalues.size())) {
    const auto& method_tuple = bytecode_ivalues.at(i).toTupleRef().elements();
    auto type_table_tuple =
        method_tuple.at(1).toTupleRef().elements()[BYTECODE_INDEX_TYPE];
    const auto& type_table =
        type_table_tuple.toTupleRef().elements()[1].toTupleRef().elements();

    // type_table is a list of IValue, and each IValue is a string,
    // for example: "Dict[int, Tuple[Tensor, Tensor, Tensor]]"
    std::vector<std::string> type_name_list;
    for (const auto& type_definition : type_table) {
      std::string type_name = type_definition.toStringRef();
      type_name_list.emplace_back(type_name);
    }
    at::TypeParser parser(type_name_list);
    parser.parseList();
    contained_types = parser.getContainedTypes();
  }

  return contained_types;
}

/********************** Compatibility Checker **********************/

ModelCompatibilityInfo ModelCompatibilityInfo::get(std::istream& in) {
  std::unique_ptr<IStreamAdapter> rai = std::make_unique<IStreamAdapter>(&in);
  return get(std::move(rai));
}

ModelCompatibilityInfo ModelCompatibilityInfo::get(
    const std::string& filename) {
  std::unique_ptr<FileAdapter> rai = std::make_unique<FileAdapter>(filename);
  return get(std::move(rai));
}

ModelCompatibilityInfo ModelCompatibilityInfo::get(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai) {
  if (!check_zip_file(rai)) {
    TORCH_CHECK(
        false, "Failed to open zip file for model compatibility information");
  }
  PyTorchStreamReader reader(std::move(rai));
  std::vector<IValue> bytecode_values = get_bytecode_ivalues(reader);
  uint64_t model_bytecode_version =
      _get_model_bytecode_version(bytecode_values);
  auto model_info = _get_model_ops_and_info(bytecode_values);
  std::unordered_set<std::string> type_table =
      _get_mobile_model_contained_types(bytecode_values);
  uint64_t operator_version = _get_model_operator_version(reader);
  return ModelCompatibilityInfo{
      model_bytecode_version, model_info, type_table, operator_version};
}

ModelCompatCheckResult is_compatible(
    RuntimeCompatibilityInfo runtime_info,
    const ModelCompatibilityInfo& model_info) {
  ModelCompatCheckResult result = {ModelCompatibilityStatus::OK, {}};
  // Check that the models bytecode version is less than or equal to
  // kMaxSupportedBytecodeVersion from the runtime
  if (model_info.bytecode_version >
      runtime_info.min_max_supported_bytecode_version.second) {
    result.status = ModelCompatibilityStatus::ERROR;
    std::ostringstream s;
    s << "model bytecode version " << model_info.bytecode_version
      << "is greater than the max supported bytecode version in runtimes "
      << runtime_info.min_max_supported_bytecode_version.second;
    result.errors.emplace_back(s.str());
  } else if (
      model_info.bytecode_version <
      runtime_info.min_max_supported_bytecode_version.first) {
    result.status = ModelCompatibilityStatus::ERROR;
    std::ostringstream s;
    s << "model bytecode version " << model_info.bytecode_version
      << "is less than the minimum supported bytecode version in runtime "
      << runtime_info.min_max_supported_bytecode_version.first;
    result.errors.emplace_back(s.str());
  }

  std::unordered_set<std::string> supported_type = runtime_info.supported_types;

  // Check type table
  for (const auto& type_name : model_info.type_table) {
    if (supported_type.find(type_name) == supported_type.end()) {
      result.status = ModelCompatibilityStatus::ERROR;
      std::ostringstream s;
      s << "Primitive type: '" << type_name
        << "' is not supported in current runtime";
      result.errors.push_back(s.str());
    }
  }

  // Check operators
  std::unordered_map<std::string, OperatorInfo> operator_info =
      model_info.operator_info;
  for (auto const& op : operator_info) {
    std::string op_name = op.first;
    OperatorInfo model_op_info = op.second;

    // Check if operator not present in runtime
    if (runtime_info.operator_info.find(op_name) ==
        runtime_info.operator_info.end()) {
      result.status = ModelCompatibilityStatus::ERROR;
      std::ostringstream s;
      s << "Operator '" << op_name << "' missing from runtime (not found)";
      result.errors.push_back(s.str());
    } else {
      OperatorInfo runtime_op_info = runtime_info.operator_info.at(op_name);

      // If the runtime op has no schema information its a false alarm and isn't
      // actually useable
      if (!runtime_op_info.num_schema_args.has_value()) {
        result.status = ModelCompatibilityStatus::ERROR;
        std::ostringstream s;
        s << "Operator '" << op_name
          << "' missing from runtime (missing schema)";
        result.errors.push_back(s.str());
      } else {
        // Check if the model operator has schema information. If it doesn't
        // then the model is from a bytecode version < 6 and we are done. If the
        // model has more args than the runtime, then the runtime can't know
        // what to do so we aren't compatible. If the runtime has more args than
        // the model then we can just use default values and be fine.
        if (model_op_info.num_schema_args.has_value() &&
            (model_op_info.num_schema_args.value() >
             runtime_op_info.num_schema_args.value())) {
          result.status = ModelCompatibilityStatus::ERROR;
          std::ostringstream s;
          s << "Operator schema for'" << op_name << "' has "
            << model_op_info.num_schema_args.value()
            << " args in model but only "
            << runtime_op_info.num_schema_args.value() << " in the runtime";
          result.errors.push_back(s.str());
        }
      }
    }
  }

  // Check Operator Versions
  if (model_info.operator_version <
          runtime_info.min_max_supported_opperator_versions.first ||
      model_info.operator_version >
          runtime_info.min_max_supported_opperator_versions.second) {
    result.status = ModelCompatibilityStatus::ERROR;
    std::ostringstream s;
    s << "Model Operator Version " << model_info.operator_version
      << "is not within supported version range of the runtime "
      << runtime_info.min_max_supported_opperator_versions.first << " to "
      << runtime_info.min_max_supported_opperator_versions.second;
    result.errors.push_back(s.str());
  }

  return result;
}

} // namespace torch::jit

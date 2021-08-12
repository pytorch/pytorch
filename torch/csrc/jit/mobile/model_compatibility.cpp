#include <ATen/core/ivalue.h>
#include <caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/api/compilation_unit.h> // removed after using simple type_resolver/obj_loader
#include <torch/csrc/jit/mobile/import.h> // removed after using simple type_resolver/obj_loader
#include <torch/csrc/jit/mobile/model_compatibility.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/serialization/import_read.h>

#include <sstream>
#include <string>
#include <vector>

namespace c10 {
TypePtr parseType(const std::string& pythonStr);
} // namespace c10

namespace torch {
namespace jit {

using caffe2::serialize::FileAdapter;
using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::ReadAdapterInterface;
struct SupportedType;

c10::IValue readArchive(
    const std::string& archive_name,
    PyTorchStreamReader& stream_reader) {
  c10::optional<at::Device> device;
  std::shared_ptr<CompilationUnit> compilation_unit =
      std::make_shared<CompilationUnit>();

  // TODO (T90180710): Simplify type_resolver and obj_loader when getting
  // bytecode version from model
  auto type_resolver = [&](const c10::QualifiedName& qn) {
    return typeResolverMobile(qn, compilation_unit);
  };

  std::shared_ptr<mobile::CompilationUnit> mobile_compilation_unit =
      std::make_shared<mobile::CompilationUnit>();
  auto obj_loader = [&](at::StrongTypePtr type, IValue input) {
    return objLoaderMobile(type, input, mobile_compilation_unit);
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
      stream_reader);
  return ivalues;
}

std::vector<IValue> get_bytecode_ivalues(PyTorchStreamReader& reader) {
  std::vector<IValue> bytecode_values;
  bytecode_values = readArchive("bytecode", reader).toTuple()->elements();
  return bytecode_values;
}

/********************** Bytecode **********************/

// Forward declare
uint64_t _get_model_bytecode_version(
    const std::vector<IValue>& bytecode_ivalues);

uint64_t _get_model_bytecode_version(std::istream& in) {
  std::unique_ptr<IStreamAdapter> rai = std::make_unique<IStreamAdapter>(&in);
  return _get_model_bytecode_version(std::move(rai));
}

uint64_t _get_model_bytecode_version(const std::string& filename) {
  std::unique_ptr<FileAdapter> rai = std::make_unique<FileAdapter>(filename);
  return _get_model_bytecode_version(std::move(rai));
}

uint64_t _get_model_bytecode_version(
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
    auto method_tuple = bytecode_ivalues.at(i).toTuple()->elements();
    auto operators_tuple = method_tuple.at(1).toTuple()->elements()[1];
    auto operators = operators_tuple.toTuple()->elements()[1];
    for (auto& op_tuple : operators.toTuple()->elements()) {
      auto op = op_tuple.toTuple()->elements();

      // grab name
      std::string op_name = op.at(0).toStringRef();
      std::string op_overload_name = op.at(1).toStringRef();
      if (op_overload_name != "") {
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

// Get deduplicate type table given bytecode,
std::vector<IValue> _get_type_table(std::vector<IValue> bytecode_ivalues) {
  std::vector<IValue> all_type_table;
  // type table can be either string (primitive type) or tuple (custom type)
  // IValue don't have a good hash function for IValue, so use type name
  // (string) as the hash.
  std::unordered_set<std::string> all_type_names;
  for (const auto i : c10::irange(1, bytecode_ivalues.size())) {
    auto method_tuple = bytecode_ivalues.at(i).toTuple()->elements();
    auto type_table_tuple = method_tuple.at(1).toTuple()->elements()[3];
    auto type_table =
        type_table_tuple.toTuple()->elements()[1].toTuple()->elements();
    for (const auto& type_definition : type_table) {
      std::string type_name;
      if (type_definition.isString()) {
        type_name = type_definition.toString()->string();
      } else if (type_definition.isTuple()) {
        type_name =
            type_definition.toTuple()->elements()[0].toString()->string();
      }
      if (all_type_names.find(type_name) == all_type_names.end()) {
        all_type_names.insert(type_name);
        all_type_table.emplace_back(type_definition);
      }
    }
  }

  return all_type_table;
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
  auto bytecode_values = get_bytecode_ivalues(reader);
  uint64_t model_bytecode_version =
      _get_model_bytecode_version(bytecode_values);
  auto model_info = _get_model_ops_and_info(bytecode_values);
  std::vector<IValue> type_table = _get_type_table(bytecode_values);
  return ModelCompatibilityInfo{model_bytecode_version, model_info, type_table};
}

std::unordered_set<std::string> splitString(const std::string& str) {
  std::unordered_set<std::string> output;
  std::string::size_type start = 0;
  std::string delimiters = "[], ";
  std::string::size_type last = str.find_first_of(delimiters);

  // keep searching another delimeters until the end of the string.
  while (last != std::string::npos) {
    // if last is greater, one token is ready
    if (last > start) {
      output.insert(str.substr(start, last - start));
    }
    // reset start to the first character of the next token
    start = ++last;
    // find the first space and we start searchingat the first character of the
    // next word
    last = str.find_first_of(delimiters, last);
  }

  // pick up the last token when not reaching the end
  if (start < str.length()) {
    output.insert(str.substr(start));
  }
  return output;
}

SupportedType possible_types(IValue type_ivalue) {
  std::unordered_set<std::string> possible_primitive_types;
  std::unordered_set<std::string> possible_custom_types;
  if (type_ivalue.isString()) {
    std::string type_name = type_ivalue.toString()->string();
    // Convert a type_name to a list of primitive type, for example:
    // Dict[int, Tuple[Tensor, Tensor, Tensor]] => [Dict, Tuple, Tensor]
    auto primitive_types = splitString(type_name);
    possible_primitive_types.insert(
        primitive_types.begin(), primitive_types.end());
  }
  return SupportedType{possible_primitive_types, possible_custom_types};
}

ModelCompatCheckResult is_compatible(
    RuntimeCompatibilityInfo runtime_info,
    ModelCompatibilityInfo model_info) {
  ModelCompatCheckResult result = {ModelCompatibilityStatus::OK, {}};
  // Check that the models bytecode version is less than or equal to
  // kMaxSupportedBytecodeVersion from the runtime
  if (model_info.bytecode_version > runtime_info.bytecode_version) {
    result.status = ModelCompatibilityStatus::ERROR;
    std::ostringstream s;
    s << "model bytecode version " << model_info.bytecode_version
      << "is greater than the runtimes " << runtime_info.bytecode_version;
    result.errors.emplace_back(s.str());
  }

  SupportedType supported_type = _get_supported_types();
  std::unordered_set<std::string> all_primitive_types;
  std::unordered_set<std::string> all_custom_types;

  // Check type table
  for (auto const& type : model_info.type_table) {
    auto possible_type_list = possible_types(type);
    all_primitive_types.insert(
        possible_type_list.primitive_types.begin(),
        possible_type_list.primitive_types.end());
    all_custom_types.insert(
        possible_type_list.custom_types.begin(),
        possible_type_list.custom_types.end());
  }

  for (const auto& primitive_type : all_primitive_types) {
    if (supported_type.primitive_types.find(primitive_type) ==
        supported_type.primitive_types.end()) {
      result.status = ModelCompatibilityStatus::ERROR;
      std::ostringstream s;
      s << "Primitive type: '" << primitive_type
        << "' is not support in current runtime";
      result.errors.push_back(s.str());
    }
  }
  for (const auto& custom_type : all_custom_types) {
    if (supported_type.custom_types.find(custom_type) ==
        supported_type.custom_types.end()) {
      result.status = ModelCompatibilityStatus::ERROR;
      std::ostringstream s;
      s << "Custom type: '" << custom_type
        << "' is not support in current runtime";
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
  return result;
}

} // namespace jit
} // namespace torch

#include <ATen/core/ivalue.h>
#include <caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/api/compilation_unit.h> // removed after using simple type_resolver/obj_loader
#include <torch/csrc/jit/mobile/import.h> // removed after using simple type_resolver/obj_loader
#include <torch/csrc/jit/mobile/model_compatibility.h>
#include <torch/csrc/jit/serialization/import_read.h>

#include <sstream>
#include <string>
#include <vector>

namespace torch {
namespace jit {

using caffe2::serialize::FileAdapter;
using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::ReadAdapterInterface;

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
  return ModelCompatibilityInfo{model_bytecode_version, model_info};
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

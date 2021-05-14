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

std::vector<IValue> get_bytecode_values(PyTorchStreamReader& reader) {
  std::vector<IValue> bytecode_values;
  bytecode_values = readArchive("bytecode", reader).toTuple()->elements();
  return bytecode_values;
}

// Forward declare
int64_t _get_model_bytecode_version(
    const std::vector<IValue>& bytecode_ivalues);

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
    TORCH_WARN(
        "The input model might not be generated from _save_for_mobile()");
    return -1;
  }
  PyTorchStreamReader reader(std::move(rai));
  auto bytecode_values = get_bytecode_values(reader);
  return _get_model_bytecode_version(bytecode_values);
}

int64_t _get_model_bytecode_version(
    const std::vector<IValue>& bytecode_ivalues) {
  if (!bytecode_ivalues.empty() && bytecode_ivalues[0].isInt()) {
    int64_t model_version = bytecode_ivalues[0].toInt();
    return model_version;
  }
  TORCH_WARN("Fail to get bytecode version.");
  return -1;
}

} // namespace jit
} // namespace torch

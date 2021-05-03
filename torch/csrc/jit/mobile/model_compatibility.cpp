#include <ATen/core/ivalue.h>
#include <caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/mobile/dummy_pickler.h>
#include <torch/csrc/jit/mobile/model_compatibility.h>
#include <torch/csrc/jit/serialization/import_read.h>

#include <torch/csrc/jit/api/compilation_unit.h>

#include <string>
#include <vector>

namespace torch {
namespace jit {

using caffe2::serialize::FileAdapter;
using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::ReadAdapterInterface;

namespace {

at::TypePtr resolveTypeName(const c10::QualifiedName& qn) {
  return std::make_shared<Dummy>();
}

c10::IValue readArchive(
    const std::string& archive_name,
    PyTorchStreamReader& stream_reader) {
  c10::optional<at::Device> device;
  auto compilation_unit = std::make_shared<CompilationUnit>();
  auto type_resolver = [&](const c10::QualifiedName& qn) {
    return c10::StrongTypePtr(compilation_unit, resolveTypeName(qn));
  };
  auto ivalues = torch::jit::readArchiveAndTensors(
      archive_name, type_resolver, nullptr, device, stream_reader);
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
    return true;
  }
  return false;
}

std::vector<IValue> get_bytecode_values(PyTorchStreamReader& reader) {
  std::vector<IValue> bytecode_values;
  bytecode_values = readArchive("bytecode", reader).toTuple()->elements();
  return bytecode_values;
}

} // namespace

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
  if (check_zip_file(rai)) {
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

} // namespace jit
} // namespace torch

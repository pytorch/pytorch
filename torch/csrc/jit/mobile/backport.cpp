#include <ATen/core/ivalue.h>
#include <caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/mobile/backport.h>
#include <torch/csrc/jit/mobile/backport_factory.h>
#include <torch/csrc/jit/mobile/model_compatibility.h>

#include <string>

namespace torch {
namespace jit {

using caffe2::serialize::FileAdapter;
using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::PyTorchStreamWriter;
using caffe2::serialize::ReadAdapterInterface;

// Forward declare so that _backport_for_mobile() overloads can
// call this method directly.
bool _backport_for_mobile_impl(
    std::shared_ptr<ReadAdapterInterface> rai,
    PyTorchStreamWriter& writer);

bool _backport_for_mobile(std::istream& in, std::ostream& out) {
  std::unique_ptr<IStreamAdapter> rai = std::make_unique<IStreamAdapter>(&in);
  auto writer_func = [&](const void* buf, size_t nbytes) -> size_t {
    out.write(static_cast<const char*>(buf), nbytes);
    return !out ? 0 : nbytes;
  };
  PyTorchStreamWriter writer(writer_func);
  return _backport_for_mobile(std::move(rai), writer);
}

bool _backport_for_mobile(
    std::istream& in,
    const std::string& output_filename) {
  std::unique_ptr<IStreamAdapter> rai = std::make_unique<IStreamAdapter>(&in);
  PyTorchStreamWriter writer(output_filename);
  return _backport_for_mobile(std::move(rai), writer);
}

bool _backport_for_mobile(
    const std::string& input_filename,
    std::ostream& out) {
  std::unique_ptr<FileAdapter> rai =
      std::make_unique<FileAdapter>(input_filename);
  auto writer_func = [&](const void* buf, size_t nbytes) -> size_t {
    out.write(static_cast<const char*>(buf), nbytes);
    return !out ? 0 : nbytes;
  };
  PyTorchStreamWriter writer(writer_func);
  return _backport_for_mobile_impl(std::move(rai), writer);
}

bool _backport_for_mobile(
    const std::string& input_filename,
    const std::string& output_filename) {
  std::unique_ptr<FileAdapter> rai =
      std::make_unique<FileAdapter>(input_filename);
  PyTorchStreamWriter writer(output_filename);
  return _backport_for_mobile_impl(std::move(rai), writer);
}

bool _backport_for_mobile(
    std::shared_ptr<ReadAdapterInterface> rai,
    PyTorchStreamWriter& writer) {
  return _backport_for_mobile_impl(std::move(rai), writer);
}

bool _backport_for_mobile_impl(
    std::shared_ptr<ReadAdapterInterface> rai,
    PyTorchStreamWriter& writer) {
  if (check_zip_file(rai)) {
    return false;
  }

  PyTorchStreamReader reader(std::move(rai));
  std::vector<c10::IValue> bytecode_values = get_bytecode_values(reader);
  int64_t bytecode_version = _get_model_bytecode_version(bytecode_values);
  auto to_bytecode_version = bytecode_version - 1;
  if (to_bytecode_version == kBytecodeVersionV4) {
    return backport_v5_to_v4(reader, writer, bytecode_values);
  }
  TORCH_WARN(
      "Backport doesn't support backport to version ", to_bytecode_version);
  return false;
}

} // namespace jit
} // namespace torch

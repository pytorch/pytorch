#include <ATen/core/ivalue.h>
#include <caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/mobile/backport.h>
#include <torch/csrc/jit/mobile/backport_manager.h>
#include <torch/csrc/jit/mobile/model_compatibility.h>

#include <string>

namespace torch {
namespace jit {

using caffe2::serialize::FileAdapter;
using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::PyTorchStreamWriter;
using caffe2::serialize::ReadAdapterInterface;

const static BackportManager backportManager;

// Forward declare so that _backport_for_mobile() overloads can
// call this method directly.
bool _backport_for_mobile_impl(
    std::shared_ptr<IStreamAdapter> istream_adapter,
    PyTorchStreamWriter& writer,
    const int64_t to_version);

bool _backport_for_mobile(
    std::istream& in,
    std::ostream& out,
    const int64_t to_version) {
  std::unique_ptr<IStreamAdapter> rai = std::make_unique<IStreamAdapter>(&in);
  auto writer_func = [&](const void* buf, size_t nbytes) -> size_t {
    out.write(static_cast<const char*>(buf), nbytes);
    return !out ? 0 : nbytes;
  };
  PyTorchStreamWriter writer(writer_func);
  return _backport_for_mobile_impl(std::move(rai), writer, to_version);
}

bool _backport_for_mobile(
    std::istream& in,
    const std::string& output_filename,
    const int64_t to_version) {
  std::unique_ptr<IStreamAdapter> istream_adapter =
      std::make_unique<IStreamAdapter>(&in);
  PyTorchStreamWriter writer(output_filename);
  return _backport_for_mobile_impl(
      std::move(istream_adapter), writer, to_version);
}

bool _backport_for_mobile(
    const std::string& input_filename,
    std::ostream& out,
    const int64_t to_version) {
  std::ifstream file_stream;
  std::unique_ptr<IStreamAdapter> istream_adapter;
  file_stream.open(input_filename, std::ifstream::in | std::ifstream::binary);
  if (!file_stream) {
    AT_ERROR("open file failed, file path: ", input_filename);
  }
  istream_adapter = std::make_unique<IStreamAdapter>(&file_stream);

  auto writer_func = [&](const void* buf, size_t nbytes) -> size_t {
    out.write(static_cast<const char*>(buf), nbytes);
    return !out ? 0 : nbytes;
  };
  PyTorchStreamWriter writer(writer_func);
  return _backport_for_mobile_impl(
      std::move(istream_adapter), writer, to_version);
}

bool _backport_for_mobile(
    const std::string& input_filename,
    const std::string& output_filename,
    const int64_t to_version) {
  std::ifstream file_stream;
  std::unique_ptr<IStreamAdapter> istream_adapter;
  file_stream.open(input_filename, std::ifstream::in | std::ifstream::binary);
  if (!file_stream) {
    AT_ERROR("open file failed, file path: ", input_filename);
  }
  istream_adapter = std::make_unique<IStreamAdapter>(&file_stream);

  PyTorchStreamWriter writer(output_filename);
  return _backport_for_mobile_impl(
      std::move(istream_adapter), writer, to_version);
}

bool _backport_for_mobile_impl(
    std::shared_ptr<IStreamAdapter> istream_adapter,
    PyTorchStreamWriter& writer,
    const int64_t to_version) {
  if (!backportManager.hasBytecodeBackportFunction(to_version + 1)) {
    return false;
  }
  int64_t from_version = _get_model_bytecode_version(istream_adapter);
  return backportManager.backport(
      istream_adapter, writer, from_version, to_version);
}

} // namespace jit
} // namespace torch

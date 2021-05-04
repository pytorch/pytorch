#include <c10/util/Exception.h>
#include <caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/mobile/backport_factory.h>

namespace torch {
namespace jit {

using caffe2::serialize::FileAdapter;
using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::PyTorchStreamWriter;
using caffe2::serialize::ReadAdapterInterface;

// A generic contract for backport logic to the previous bytecode version.
// Args:
// * PyTorchStreamReader has access to the input model from N bytecode version.
// * PyTorchStreamWriter has access to the output model backported to the
// previous N-1 bytecode version. Returns true if successful, false otherwise.
using BytecodeBackportFunction = std::function<bool(
    caffe2::serialize::PyTorchStreamReader&,
    caffe2::serialize::PyTorchStreamWriter&)>;

bool backport_v5_to_v4(
    caffe2::serialize::PyTorchStreamReader& rai,
    caffe2::serialize::PyTorchStreamWriter& writer);

BackportFactory::BackportFactory() {
  registerBytecodeBackportFunction(kBytecodeVersionV5, backport_v5_to_v4);
}

void BackportFactory::registerBytecodeBackportFunction(
    const int64_t from_version,
    const BytecodeBackportFunction& backport_function) {
  TORCH_CHECK(
      !hasBytecodeBackportFunction(from_version),
      "Backporting from version ",
      from_version,
      " is already registered.");
  bytecodeBackportFunctions()[from_version] = backport_function;
}

// The main function to run backport from version n to version i.
// All models (file or buffer) will be converted stream first, and
// istream_adapter has access to it. During the backport process,
// the intermediate result will be stored with stream.
bool BackportFactory::backport(
    std::shared_ptr<IStreamAdapter> istream_adapter,
    PyTorchStreamWriter& writer,
    int64_t from_version,
    int64_t to_version) const {
  int64_t bytecode_version = from_version;
  std::ostringstream out;
  auto writer_func = [&](const void* buf, size_t nbytes) -> size_t {
    out.write(static_cast<const char*>(buf), nbytes);
    return !out ? 0 : nbytes;
  };

  std::shared_ptr<IStreamAdapter> intermediate_istream_adapter =
      istream_adapter;
  std::ostringstream oss;

  while (bytecode_version > to_version) {
    // Read from intermediate writer result if ostream is not empty, otherwise
    // it means that it's the first time to backport and read from the source.
    if (!out.str().empty()) {
      std::istringstream iss(out.str());
      intermediate_istream_adapter =
          std::make_shared<caffe2::serialize::IStreamAdapter>(&iss);
    }
    out.clear();

    PyTorchStreamReader intermediate_reader(intermediate_istream_adapter);
    PyTorchStreamWriter intermediate_writer(writer_func);

    if (!hasBytecodeBackportFunction(bytecode_version)) {
      return false;
    }

    // When it's the last backport process, write to the final destination
    // otherwise, export to the intermediate ostream.
    if (bytecode_version - 1 == to_version) {
      bytecodeBackportFunctions()[bytecode_version--](
          intermediate_reader, writer);
    } else {
      bytecodeBackportFunctions()[bytecode_version--](
          intermediate_reader, intermediate_writer);
    }
  }
  return true;
}

} // namespace jit
} // namespace torch

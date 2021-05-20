#pragma once

#include <functional>
#include <memory>
#include <unordered_map>

namespace c10 {
struct IValue;
}

namespace caffe2 {
namespace serialize {
class IStreamAdapter;
class ReadAdapterInterface;
class PyTorchStreamWriter;
class PyTorchStreamReader;
} // namespace serialize
} // namespace caffe2

namespace torch {
namespace jit {

/*
BackportManager manages a list of backport from n to n-1 function, and provides
function to check if a specific function exists.
*/
class BackportManager final {
 public:
  bool hasBytecodeBackportFunction(const int64_t from_version) const;

  std::unordered_map<
      int64_t,
      std::function<bool(
          caffe2::serialize::PyTorchStreamReader&,
          caffe2::serialize::PyTorchStreamWriter&)>>&
  bytecodeBackportFunctions() const;

  bool backport(
      std::shared_ptr<caffe2::serialize::IStreamAdapter> istream_adapter,
      caffe2::serialize::PyTorchStreamWriter& final_writer,
      int64_t from_version,
      int64_t to_version) const;

  BackportManager(BackportManager const&) = delete;
  BackportManager& operator=(BackportManager const&) = delete;
  BackportManager();

 private:
  // Registry of backport functions.
  void registerBytecodeBackportFunction(
      const int64_t from_version,
      const std::function<bool(
          caffe2::serialize::PyTorchStreamReader&,
          caffe2::serialize::PyTorchStreamWriter&)>& backport_function);
};

} // namespace jit
} // namespace torch

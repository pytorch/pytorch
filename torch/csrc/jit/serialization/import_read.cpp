#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/serialization/import_read.h>

namespace torch {
namespace jit {

IValue readArchiveAndTensors(
    const std::string& archive_name,
    const std::string& pickle_prefix,
    const std::string& tensor_prefix,
    c10::optional<TypeResolver> type_resolver,
    c10::optional<ObjLoader> obj_loader,
    c10::optional<at::Device> device,
    caffe2::serialize::PyTorchStreamReader& stream_reader) {
  std::string picklename = pickle_prefix + archive_name + ".pkl";
  at::DataPtr pickle_ptr;
  size_t pickle_size = 0;
  std::tie(pickle_ptr, pickle_size) = stream_reader.getRecord(picklename);

  size_t bytes_read = 0;
  auto data = reinterpret_cast<const char*>(pickle_ptr.get());
  auto reader = [&](char* buffer, size_t len) -> size_t {
    if (bytes_read >= pickle_size) {
      return 0;
    }
    len = std::min(pickle_size - bytes_read, len);
    // Copy len bytes into buffer
    const char* start = data + bytes_read;
    std::memcpy(buffer, start, len);
    bytes_read += len;
    return len;
  };
  std::string tensor_dir_path =
      (tensor_prefix.compare("") != 0) ? tensor_prefix : archive_name + "/";
  auto read_record = [&](const std::string& name) {
    std::string ss = tensor_dir_path + name;
    return std::get<0>(stream_reader.getRecord(ss));
  };

  Unpickler unpickler(
      reader,
      type_resolver ? std::move(*type_resolver) : nullptr,
      obj_loader ? std::move(*obj_loader) : nullptr,
      std::move(read_record),
      device);
  unpickler.set_version(stream_reader.version());
  return unpickler.parse_ivalue();
}

} // namespace jit
} // namespace torch

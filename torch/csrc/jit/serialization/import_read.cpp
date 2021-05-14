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
    caffe2::serialize::PyTorchStreamReader& stream_reader,
    std::shared_ptr<StorageContext> storage_context) {
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
      device,
      false,
      storage_context);
  unpickler.set_version(stream_reader.version());
  return unpickler.parse_ivalue();
}

bool check_zip_file(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai) {
  std::array<uint8_t, 2> first_short{};
  static constexpr uint8_t first_slot = 0x80;
  static constexpr uint8_t second_slot = 0x02;
  rai->read(
      /*pos=*/0,
      /*buf=*/&first_short,
      /*n=*/2,
      /*what=*/"checking archive");

  // NB: zip files by spec can start with any data, so technically they might
  // start with 0x80 0x02, but in practice zip files start with a file entry
  // which begins with 0x04034b50. Furthermore, PyTorch will never produce zip
  // files that do not start with the file entry, so it is relatively safe to
  // perform this check.
  return !(first_short[0] == first_slot && first_short[1] == second_slot);
}

} // namespace jit
} // namespace torch

#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/serialization/import_read.h>

namespace torch {
namespace jit {

IValue readArchiveAndTensors(
    const std::string& archive_name,
    c10::optional<TypeResolver> type_resolver,
    c10::optional<ObjLoader> obj_loader,
    c10::optional<at::Device> device,
    caffe2::serialize::PyTorchStreamReader& stream_reader) {
  std::string picklename = archive_name + ".pkl";
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

  static const char slash = '/';
  auto read_record = [&](const std::string& name) {
    std::size_t found = name.find(slash);
    std::stringstream ss;
    // In bytecode version 4, the tensor root_key doesn't include the parent
    // path To support backward compatibility, when the name doesn't include
    // slash assume it's version 4 and attach the archive_name_plus_slash The
    // example tensor format is: torch._utils._rebuild_tensor_v2(
    //     pers.obj(('storage', torch.FloatStorage, '17', 'cpu', 22736),),
    //     0,
    //     (1, 464, 7, 7),
    //     (22736, 49, 7, 1),
    //     False,
    //     collections.OrderedDict())
    if (found == std::string::npos) {
      ss << archive_name << slash << name;
      return std::get<0>(stream_reader.getRecord(ss.str()));
    }

    // In bytecode version 4+, the tensor root_key in bytecode will include the
    // parent path. The example tensor format is:
    // torch._utils._rebuild_tensor_v2(
    //     pers.obj(('storage', torch.FloatStorage, 'constants/17', 'cpu',
    //     22736),), 0, (1, 464, 7, 7), (22736, 49, 7, 1), False,
    //     collections.OrderedDict())
    ss << name;
    return std::get<0>(stream_reader.getRecord(ss.str()));
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

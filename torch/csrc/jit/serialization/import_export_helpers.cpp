#include <torch/csrc/jit/serialization/import_export_helpers.h>

#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/serialization/source_range_serialization_impl.h>

#include <c10/util/Exception.h>

#include <algorithm>

namespace torch::jit {

static const std::string kExportSuffix = "py";

std::string qualifierToArchivePath(
    const std::string& qualifier,
    const std::string& export_prefix) {
  std::string path = qualifier;
  std::replace_if(
      path.begin(), path.end(), [](char c) { return c == '.'; }, '/');
  return export_prefix + path + "." + kExportSuffix;
}

std::shared_ptr<Source> findSourceInArchiveFromQualifier(
    caffe2::serialize::PyTorchStreamReader& reader,
    const std::string& export_prefix,
    const std::string& qualifier) {
  const std::string path = qualifierToArchivePath(qualifier, export_prefix);
  if (!reader.hasRecord(path)) {
    return nullptr;
  }
  auto [data, size] = reader.getRecord(path);

  std::shared_ptr<ConcreteSourceRangeUnpickler> gen_ranges = nullptr;

  std::string debug_file = path + ".debug_pkl";
  if (reader.hasRecord(debug_file)) {
    auto [debug_data, debug_size] = reader.getRecord(debug_file);
    gen_ranges = std::make_shared<ConcreteSourceRangeUnpickler>(
        std::move(debug_data), debug_size);
  }
  return std::make_shared<Source>(
      std::string(static_cast<const char*>(data.get()), size),
      path,
      1,
      gen_ranges);
}

} // namespace torch::jit

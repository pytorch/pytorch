#include <torch/csrc/jit/import_export_helpers.h>
#include <c10/util/Exception.h>

#include <algorithm>

namespace torch {
namespace jit {
namespace ImportExportHelpers {

static const std::string kExportSuffix = "py";

std::string qualifierToPath(
    const std::string& qualifier,
    const std::string& export_prefix) {
  std::string path = qualifier;
  std::replace_if(
      path.begin(), path.end(), [](char c) { return c == '.'; }, '/');
  return export_prefix + path + "." + kExportSuffix;
}

std::string pathToQualifier(
    const std::string& classPath,
    const std::string& export_prefix) {
  // strip input suffix
  const auto end = classPath.rfind(kExportSuffix);
  AT_ASSERT(end != std::string::npos);

  // strip input suffix
  size_t libs_idx = classPath.find(export_prefix);
  AT_ASSERT(libs_idx == 0);

  AT_ASSERT(classPath.size() > export_prefix.size());
  const auto start = export_prefix.size();

  std::string class_qualifier = classPath.substr(start, end - start);
  std::replace_if(
      class_qualifier.begin(),
      class_qualifier.end(),
      [](char c) { return c == '/'; },
      '.');

  return class_qualifier;
}
} // namespace ImportExportHelpers
} // namespace jit
} // namespace torch

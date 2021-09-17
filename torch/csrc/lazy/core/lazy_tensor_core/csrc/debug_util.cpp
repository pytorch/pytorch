#include "lazy_tensor_core/csrc/debug_util.h"

#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_set>

#include "lazy_tensor_core/csrc/device.h"
#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensor_core/csrc/ir_dump_util.h"
#include "lazy_tensor_core/csrc/ir_util.h"
#include "lazy_tensor_core/csrc/python_util.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/sys_util.h"
#include "lazy_tensors/computation_client/unique.h"
#include "lazy_tensors/str_split.h"

namespace torch_lazy_tensors {
namespace {

DebugUtil::GraphFormat DefaultGraphFormat() {
  std::string fmt_str =
      lazy_tensors::sys_util::GetEnvString("LTC_SAVE_TENSORS_FMT", "text");
  if (fmt_str == "text") {
    return DebugUtil::GraphFormat::kText;
  } else if (fmt_str == "backend") {
    return DebugUtil::GraphFormat::kBackend;
  } else if (fmt_str == "dot") {
    return DebugUtil::GraphFormat::kDot;
  }
  LTC_ERROR() << "Invalid save graph format: " << fmt_str;
}

std::unordered_set<std::string>* LoadExperiments() {
  std::unique_ptr<std::unordered_set<std::string>> xset =
      std::make_unique<std::unordered_set<std::string>>();
  std::string experiments =
      lazy_tensors::sys_util::GetEnvString("LTC_EXPERIMENTAL", "");
  std::vector<std::string> experiment_list =
      lazy_tensors::StrSplit(experiments, ':');
  for (auto& name : experiment_list) {
    xset->insert(name);
  }
  return xset.release();
}

}  // namespace

DebugUtil::GraphFormat DebugUtil::GetDefaultGraphFormat() {
  static GraphFormat format = DefaultGraphFormat();
  return format;
}

std::string DebugUtil::GetTensorsGraphInfo(
    lazy_tensors::Span<const LazyTensor> tensors,
    const std::vector<size_t>* indices, GraphFormat format) {
  std::vector<const ir::Node*> root_nodes;
  std::vector<ir::Value> root_values;
  std::vector<lazy_tensors::hash_t> root_hashes;
  lazy_tensors::util::Unique<Device> unique_device;
  if (indices != nullptr) {
    for (auto index : *indices) {
      const LazyTensor& tensor = tensors[index];
      ir::Value ir_value = tensor.CurrentIrValue();
      if (ir_value) {
        root_nodes.push_back(ir_value.node.get());
        root_hashes.push_back(ir_value.hash());
        root_values.push_back(std::move(ir_value));
        unique_device.set(tensor.GetDevice());
      }
    }
  } else {
    for (auto& tensor : tensors) {
      ir::Value ir_value = tensor.CurrentIrValue();
      if (ir_value) {
        root_nodes.push_back(ir_value.node.get());
        root_hashes.push_back(ir_value.hash());
        root_values.push_back(std::move(ir_value));
        unique_device.set(tensor.GetDevice());
      }
    }
  }
  std::stringstream ss;
  std::vector<SourceLocation> frames = GetPythonFrames();
  ss << "TensorsGraphInfo:\n";
  for (auto& location : frames) {
    ss << "  " << location.function << " (" << location.file << ":"
       << location.line << ")\n";
  }
  ss << "\nHashes: (";
  for (size_t i = 0; i < root_hashes.size(); ++i) {
    if (i > 0) {
      ss << ", ";
    }
    ss << lazy_tensors::util::HexHash(root_hashes[i]);
  }
  ss << ")\n";

  std::string graph_str;
  if (format == GraphFormat::kText) {
    graph_str = ir::DumpUtil::ToText(root_nodes);
  } else if (format == GraphFormat::kDot) {
    graph_str = ir::DumpUtil::ToDot(root_nodes);
  } else if (format == GraphFormat::kBackend) {
    graph_str = ir::DumpUtil::ToBackend(
        root_values, unique_device ? *unique_device : GetCurrentDevice());
  } else {
    LTC_ERROR() << "Invalid graph format: " << format;
  }
  ss << "\n## BEGIN_GRAPH\n" << graph_str << "\n## END_GRAPH\n\n";
  return ss.str();
}

void DebugUtil::SaveTensorsGraphInfo(
    const char* name, lazy_tensors::Span<const LazyTensor> tensors,
    const std::vector<size_t>* indices, GraphFormat format) {
  static const std::string save_file =
      lazy_tensors::sys_util::GetEnvOrdinalPath("LTC_SAVE_TENSORS_FILE", "");
  if (!save_file.empty()) {
    static std::mutex lock;
    std::string info = GetTensorsGraphInfo(tensors, indices, format);
    std::lock_guard<std::mutex> guard(lock);
    std::ofstream graph_file(save_file, std::ios_base::app);
    graph_file << "[" << name << "]\n" << info << "\n";
  }
}

bool DebugUtil::ExperimentEnabled(const std::string& name) {
  static const std::unordered_set<std::string>* xset = LoadExperiments();
  return xset->find(name) != xset->end();
}

}  // namespace torch_lazy_tensors

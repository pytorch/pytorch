#include <c10/util/irange.h>
#include <torch/csrc/lazy/core/debug_util.h>

#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_dump_util.h>
#include <torch/csrc/lazy/core/ir_util.h>
#include <torch/csrc/lazy/core/unique.h>

#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_set>

namespace torch {
namespace lazy {
namespace {

std::string GetEnvString(const char* name, const std::string& defval) {
  const char* env = std::getenv(name);
  return env != nullptr ? env : defval;
}

DebugUtil::GraphFormat DefaultGraphFormat() {
  std::string fmt_str = GetEnvString("LTC_SAVE_TENSORS_FMT", "text");
  if (fmt_str == "text") {
    return DebugUtil::GraphFormat::kText;
  } else if (fmt_str == "backend") {
    return DebugUtil::GraphFormat::kBackend;
  } else if (fmt_str == "dot") {
    return DebugUtil::GraphFormat::kDot;
  }
  LOG(ERROR) << "Invalid save graph format: " << fmt_str;
  return DebugUtil::GraphFormat::kText;
}

std::unordered_set<std::string>* LoadExperiments() {
  std::unique_ptr<std::unordered_set<std::string>> xset =
      std::make_unique<std::unordered_set<std::string>>();
  std::string experiments = GetEnvString("LTC_EXPERIMENTAL", "");
  std::vector<std::string> experiment_list =
      torch::lazy::StrSplit(experiments, ':');
  for (auto& name : experiment_list) {
    xset->insert(name);
  }
  return xset.release();
}

} // namespace

std::vector<SourceLocation> NoPythonFrames() {
  SourceLocation dummy_loc;
  dummy_loc.file = "No Python Frames";
  return {dummy_loc};
}

std::function<std::vector<SourceLocation>()>& GetPythonFramesFunction() {
  static std::function<std::vector<SourceLocation>()> func_ = NoPythonFrames;
  return func_;
}

DebugUtil::GraphFormat DebugUtil::GetDefaultGraphFormat() {
  static GraphFormat format = DefaultGraphFormat();
  return format;
}

std::string GetFirstUserFrameInPython() {
  std::string empty;
  if (!torch::lazy::GetPythonFramesFunction()) {
    return empty;
  }

  auto frames = torch::lazy::GetPythonFramesFunction()();

  for (auto i = frames.size(); i > 0; i--) {
    auto& loc = frames[i - 1];
    if (loc.file.find("site-packages") == std::string::npos) {
      std::stringstream ss;
      ss << loc.file << " " << loc.function << " " << loc.line;
      return ss.str();
    }
  }
  return empty;
}

std::string DebugUtil::GetTensorsGraphInfo(
    c10::ArrayRef<torch::lazy::LazyTensorPtr> tensors,
    const std::vector<size_t>* indices,
    GraphFormat format) {
  std::vector<const torch::lazy::Node*> root_nodes;
  std::vector<torch::lazy::Value> root_values;
  std::vector<torch::lazy::hash_t> root_hashes;
  torch::lazy::Unique<torch::lazy::BackendDevice> unique_device;
  if (indices != nullptr) {
    for (auto index : *indices) {
      const torch::lazy::LazyTensorPtr& tensor = tensors[index];
      torch::lazy::Value ir_value = tensor->CurrentIrValue();
      if (ir_value) {
        root_nodes.push_back(ir_value.node.get());
        root_hashes.push_back(ir_value.hash());
        root_values.push_back(std::move(ir_value));
        unique_device.set(tensor->GetDevice());
      }
    }
  } else {
    for (auto& tensor : tensors) {
      torch::lazy::Value ir_value = tensor->CurrentIrValue();
      if (ir_value) {
        root_nodes.push_back(ir_value.node.get());
        root_hashes.push_back(ir_value.hash());
        root_values.push_back(std::move(ir_value));
        unique_device.set(tensor->GetDevice());
      }
    }
  }
  std::stringstream ss;
  // Call into a function pointer that may backed by python or empty depending
  // on runtime
  std::vector<SourceLocation> frames = GetPythonFramesFunction()();
  ss << "Python Stacktrace:\n";
  for (auto& location : frames) {
    ss << "  " << location.function << " (" << location.file << ":"
       << location.line << ")\n";
  }
  ss << "\nHashes: (";
  for (const auto i : c10::irange(root_hashes.size())) {
    if (i > 0) {
      ss << ", ";
    }
    ss << torch::lazy::HashToString(root_hashes[i]);
  }
  ss << ")\n";

  std::string graph_str;
  if (format == GraphFormat::kText) {
    graph_str = torch::lazy::DumpUtil::ToText(root_nodes);
  } else if (format == GraphFormat::kDot) {
    graph_str = torch::lazy::DumpUtil::ToDot(root_nodes);
  } else if (format == GraphFormat::kBackend) {
    graph_str = torch::lazy::DumpUtil::ToBackend(
        root_values,
        unique_device ? *unique_device : torch::lazy::BackendDevice());
  } else {
    LOG(ERROR) << "Invalid graph format: " << format;
  }
  ss << "\n## BEGIN_GRAPH\n" << graph_str << "\n## END_GRAPH\n\n";
  return ss.str();
}

void DebugUtil::SaveTensorsGraphInfo(
    const char* name,
    c10::ArrayRef<torch::lazy::LazyTensorPtr> tensors,
    const std::vector<size_t>* indices,
    GraphFormat format) {
  static const std::string save_file =
      GetEnvString("LTC_SAVE_TENSORS_FILE", "");
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

} // namespace lazy
} // namespace torch

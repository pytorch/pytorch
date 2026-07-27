#pragma once

#include <string>
#include <vector>

#include <torch/csrc/lazy/core/tensor.h>

namespace torch::lazy {

TORCH_API std::function<std::vector<SourceLocation>()>&
GetPythonFramesFunction();

TORCH_API std::string GetFirstUserFrameInPython();

class TORCH_API DebugUtil {
 public:
  enum GraphFormat {
    kText,
    kDot,
    kBackend,
  };

  static GraphFormat GetDefaultGraphFormat();

  // Dumps the current Python frame and the IR Graph whose roots are the IR
  // values held at the tensors. If indices is not nullptr, it selects the
  // indices of the tensors whose graph will be emitted.
  static std::string GetTensorsGraphInfo(
      c10::ArrayRef<torch::lazy::LazyTensorPtr> tensors,
      const std::vector<size_t>* indices,
      GraphFormat format = GetDefaultGraphFormat());

  // If the environment variable LTC_SAVE_TENSORS_FILE is set to the proper
  // output path, an instance of the report returned by GetTensorsGraphInfo() is
  // saved.
  static void SaveTensorsGraphInfo(
      const char* name,
      c10::ArrayRef<torch::lazy::LazyTensorPtr> tensors,
      const std::vector<size_t>* indices,
      GraphFormat format = GetDefaultGraphFormat());

  static bool ExperimentEnabled(const std::string& name);
};

} // namespace torch::lazy

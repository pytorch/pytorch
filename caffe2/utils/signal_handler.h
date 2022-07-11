#pragma once

#include <c10/util/signal_handler.h>

namespace caffe2 {

#if defined(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)
class TORCH_API C2FatalSignalHandler : public c10::FatalSignalHandler {
 public:
  void fatalSignalHandlerPostProcess() override;
  static C2FatalSignalHandler& getInstance();

 private:
  explicit C2FatalSignalHandler();
};

// This works by setting up certain fatal signal handlers. Previous fatal
// signal handlers will still be called when the signal is raised. Defaults
// to being off.
TORCH_API void setPrintStackTracesOnFatalSignal(bool print);
TORCH_API bool printStackTracesOnFatalSignal();
#endif // defined(C10_SUPPORTS_FATAL_SIGNAL_HANDLER)

} // namespace caffe2

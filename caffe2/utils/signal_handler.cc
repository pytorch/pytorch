#include "caffe2/utils/signal_handler.h"
#include "caffe2/core/init.h"
#include "caffe2/core/workspace.h"

namespace {

// NOLINTNEXTLINE(clang-diagnostic-unused-function)
void printBlobSizes() {
  ::caffe2::Workspace::ForEach(
      [&](::caffe2::Workspace* ws) { ws->PrintBlobSizes(); });
}

} // namespace

#if defined(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)
C10_DEFINE_bool(
    caffe2_print_stacktraces,
    false,
    "If set, prints stacktraces when a fatal signal is raised.");

namespace caffe2 {

C2FatalSignalHandler::C2FatalSignalHandler() {}

C2FatalSignalHandler& C2FatalSignalHandler::getInstance() {
  // Leaky singleton to avoid module destructor race.
  static C2FatalSignalHandler* handler = new C2FatalSignalHandler();
  return *handler;
}

void C2FatalSignalHandler::fatalSignalHandlerPostProcess() {
  printBlobSizes();
}

void setPrintStackTracesOnFatalSignal(bool print) {
  C2FatalSignalHandler::getInstance().setPrintStackTracesOnFatalSignal(print);
}

bool printStackTracesOnFatalSignal() {
  return C2FatalSignalHandler::getInstance().printStackTracesOnFatalSignal();
}

namespace internal {
bool Caffe2InitFatalSignalHandler(int*, char***) {
  if (FLAGS_caffe2_print_stacktraces) {
    setPrintStackTracesOnFatalSignal(true);
  }
  return true;
}

REGISTER_CAFFE2_INIT_FUNCTION(
    Caffe2InitFatalSignalHandler,
    &Caffe2InitFatalSignalHandler,
    "Inits signal handlers for fatal signals so we can see what if"
    " caffe2_print_stacktraces is set.");

} // namespace internal
} // namespace caffe2
#endif // defined(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)

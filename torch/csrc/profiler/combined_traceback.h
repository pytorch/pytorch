#pragma once

#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/profiler/unwind/unwind.h>

#include <optional>
#include <string>
#include <vector>

namespace torch {

// struct that holds the result of symbolizing multiple tracebacks
// each traceback is a list of indices into all_frames
// (lots of Frames get duplicated across traces)
struct TORCH_API SymbolizedTracebacks {
  std::vector<unwind::Frame> all_frames;
  // index into all_frames, so that
  // it is possible to dedupe frame objects in
  // construction of python objects
  std::vector<std::vector<uint64_t>> tracebacks;
};

struct TORCH_API CapturedTraceback : public c10::GatheredContext {
  struct PyFrame {
    void* code; // PyCodeObject*, but python headers not present
    int lasti;
  };

  static std::shared_ptr<CapturedTraceback> gather(
      bool python,
      bool script,
      bool cpp);
  CapturedTraceback() = default;
  CapturedTraceback(const CapturedTraceback&) = delete;
  CapturedTraceback& operator=(const CapturedTraceback&) = delete;
  CapturedTraceback(CapturedTraceback&&) noexcept = default;
  CapturedTraceback& operator=(CapturedTraceback&&) noexcept = delete;
  ~CapturedTraceback() override;

  using visitproc = int (*)(void* self, void* arg);

  struct Python {
    // Check if it's safe to gather Python frames from the current thread.
    // Returns false for pure C++ threads that cannot acquire the GIL.
    virtual bool canGather() = 0;
    virtual std::vector<PyFrame> gather() = 0;
    virtual void release(std::vector<PyFrame>& frames) = 0;
    virtual void appendSymbolized(
        const std::vector<PyFrame>& to_symbolize,
        SymbolizedTracebacks& st) = 0;
    // tp_traverse/tp_clear implementations
    virtual int traverse(
        std::vector<PyFrame>& frames,
        visitproc visit,
        void* arg) = 0;
    virtual int clear(std::vector<PyFrame>& frames) = 0;
    // Gather forward traceback from the current autograd node's anomaly
    // metadata. Returns a vector of strings representing the forward stack
    // trace, or empty if not available.
    virtual std::vector<std::string> gatherForwardTraceback() {
      return {};
    }
    virtual ~Python() = default;
    Python* next_ = nullptr;
  };
  // called once by each python interpreter to
  // register python stack recording functionality
  // p cannot be deleted once added.
  static void addPythonUnwinder(Python* p);

  int traversePython(visitproc visit, void* arg);
  int clearPython();

 private:
  std::vector<PyFrame> frames_;
  std::vector<void*> cpp_frames_;
  std::vector<jit::StackEntry> script_frames_;
  friend TORCH_API SymbolizedTracebacks
  symbolize(const std::vector<CapturedTraceback*>& to_symbolize);

  // non-owning reference to one of the immortal Python* objects
  // registered above.
  Python* python_ = nullptr;

  // Optional forward traceback from anomaly mode.
  // This is a list of Python strings representing the forward stack trace
  // when the autograd Node was created. Used to correlate backward allocations
  // with forward operations.
  std::optional<std::vector<std::string>> forward_traceback_;

 public:
  // Set the forward traceback from anomaly mode metadata
  void set_forward_traceback(std::vector<std::string> traceback) {
    forward_traceback_ = std::move(traceback);
  }

  // Get the forward traceback if available
  const std::optional<std::vector<std::string>>& forward_traceback() const {
    return forward_traceback_;
  }
};

TORCH_API SymbolizedTracebacks
symbolize(const std::vector<CapturedTraceback*>& to_symbolize);

inline CapturedTraceback* getCapturedTracebackFromContext(
    const std::shared_ptr<c10::GatheredContext>& x) {
  auto* traceback = dynamic_cast<CapturedTraceback*>(x.get());
  TORCH_CHECK(
      traceback, "attempting to gather stack context from the wrong type.");
  return traceback;
}

} // namespace torch

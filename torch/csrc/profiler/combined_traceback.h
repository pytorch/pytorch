#pragma once

#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/profiler/unwind/unwind.h>

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
  ~CapturedTraceback();

  using visitproc = int (*)(void* self, void* arg);

  struct Python {
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
};

TORCH_API SymbolizedTracebacks
symbolize(const std::vector<CapturedTraceback*>& to_symbolize);

} // namespace torch

#pragma once

#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/profiler/unwind/unwind.h>

struct PyCodeObject;

namespace torch {

struct TORCH_API SymbolizedTracebacks {
  std::vector<unwind::Frame> all_frames;
  // index into all_frames, so that
  // it is possible to dedupe frame objects in
  // construction of python objects
  std::vector<std::vector<uint64_t>> tracebacks;
};

struct TORCH_API CapturedTraceback : public c10::GatheredContext {
  struct PyFrame {
    PyCodeObject* code;
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
  struct Python {
    virtual std::vector<PyFrame> gather() = 0;
    virtual void release(std::vector<PyFrame>& frames) = 0;
    virtual void appendSymbolized(
        const std::vector<PyFrame>& to_symbolize,
        SymbolizedTracebacks& st) = 0;
    virtual ~Python() = default;
    Python* next_ = nullptr;
  };
  static void addPythonUnwinder(Python* p);

 private:
  std::vector<PyFrame> frames_;
  std::vector<void*> cpp_frames_;
  std::vector<jit::StackEntry> script_frames_;
  friend SymbolizedTracebacks symbolize(
      const std::vector<CapturedTraceback*>& to_symbolize);
  Python* python_ = nullptr;
};

TORCH_API SymbolizedTracebacks
symbolize(const std::vector<CapturedTraceback*>& to_symbolize);

} // namespace torch

#pragma once

#include <atomic>
#include <memory>

#include <torch/csrc/jit/runtime/interpreter/code_impl.h>
#include <torch/csrc/jit/runtime/profiling_record.h>

namespace torch::jit::interpreter {

// A Frame captures function's state
// (e.g. `pc` and `base_pointer`)
// Each Frame corresponds to a call to a `Frame::function`
// which has not yet returned
// The arguments for `Frame::function`
// are located at [base_pointer + arg_number]
struct Frame {
  std::shared_ptr<CodeImpl> function;
  // program counter corresponds to the index
  // of the currently executed instruction
  size_t pc;
  // marks the start index of the frame
  // base_pointer is used by TAIL_CALL
  // to replace the current frame
  // with a frame of a bailout graph
  size_t base_pointer;

  // unique to every frame with prim::profile across all threads
  c10::optional<size_t> id;

  // RecordFunction object associated with this frame
  std::unique_ptr<at::RecordFunction> record_function;

  // symbol table for a frame
  ShapeSymbolTable symbols2dims;

  static size_t genId();
};

} // namespace torch::jit::interpreter

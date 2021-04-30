#pragma once

#include <chrono>
#include <map>
#include <string>

#include <c10/macros/Macros.h>
#include <torch/csrc/jit/frontend/source_ref.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace profiling {

struct Datapoint {
  using Timepoint = std::chrono::time_point<std::chrono::steady_clock>;
  SourceRange sourceRange;
  Timepoint start;
  Timepoint end;

 private:
  friend class InstructionSpan;
  Datapoint(SourceRange sr)
      : sourceRange(std::move(sr)), start(std::chrono::steady_clock::now()) {}
};

class TORCH_API InstructionSpan {
 public:
  C10_NODISCARD static c10::optional<InstructionSpan> tryMake(Node&);
  InstructionSpan(InstructionSpan&&);
  ~InstructionSpan();

 private:
  InstructionSpan(Node&);

  bool empty_{false};
  Datapoint datapoint_;
};

} // namespace profiling

struct TORCH_API InstructionStats {
  size_t count{0};
  std::chrono::nanoseconds duration{0};
};

class TORCH_API ScriptProfile {
  // Aggregates datapoints by function source id, then by line number.
  using LineMap = std::map<size_t, InstructionStats>;
  using Stats = std::map<SourceRef, LineMap, std::less<>>;

 public:
  void enable();
  void disable();
  const Stats& dumpStats();
  void addDatapoint(profiling::Datapoint);
  ~ScriptProfile();

 private:
  bool enabled_{false};
  std::vector<profiling::Datapoint> datapoints_;
  Stats stats_;
};

} // namespace jit
} // namespace torch

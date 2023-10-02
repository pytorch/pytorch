#pragma once

#include <chrono>
#include <map>
#include <string>

#include <ATen/core/ivalue.h>
#include <c10/macros/Macros.h>
#include <torch/csrc/jit/frontend/source_ref.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {
namespace profiling {

struct Datapoint {
  using Timepoint = std::chrono::time_point<std::chrono::steady_clock>;
  SourceRange sourceRange;
  Timepoint start;
  Timepoint end;

  explicit Datapoint(SourceRange sr)
      : sourceRange(std::move(sr)), start(std::chrono::steady_clock::now()) {}
};

class TORCH_API InstructionSpan {
 public:
  explicit InstructionSpan(Node&);
  ~InstructionSpan();
  InstructionSpan(InstructionSpan&&) = delete;
  InstructionSpan& operator=(InstructionSpan&&) = delete;

 private:
  std::unique_ptr<Datapoint> datapoint_;
};

} // namespace profiling

struct TORCH_API InstructionStats : public CustomClassHolder {
  int64_t count{0};
  std::chrono::nanoseconds duration{0};
};

class TORCH_API SourceStats : public CustomClassHolder {
 public:
  using LineMap = c10::Dict<int64_t, c10::intrusive_ptr<InstructionStats>>;

  SourceStats(SourceRef source, LineMap lineMap)
      : source_(std::move(source)), lineMap_(std::move(lineMap)) {}

  const SourceRef& getSourceRef() const {
    return source_;
  }

  const LineMap& getLineMap() const {
    return lineMap_;
  }

 private:
  SourceRef source_;
  LineMap lineMap_;
};

/**
 * ScriptProfile is an underlying C++ implementation for TorchScript profiling.
 * The profiling section is specified by calling enable() and disable():
 *
 * ...
 * scriptProfile.enable();
 * ...
 * (scripts)
 * ...
 * scriptProfile.disable();
 * ...
 *
 * To retrieve collected runtime data, users may call dumpStats() and do
 * arbitrary filtering on the data they want. Note that dumpStats() should
 * not be called inside a profiling section.
 * In general, stats are aggregated per source function body, and then by line
 * number.
 */
class TORCH_API ScriptProfile : public CustomClassHolder {
  // Aggregates datapoints by function source id, then by line number.
  using LineMap = std::map<int64_t, InstructionStats>;
  using SourceMap = std::map<SourceRef, LineMap, std::less<>>;

 public:
  void enable();
  void disable();
  const SourceMap& dumpStats();
  void addDatapoint(std::shared_ptr<profiling::Datapoint>);
  ~ScriptProfile() override;

 private:
  bool enabled_{false};
  std::vector<std::shared_ptr<profiling::Datapoint>> datapoints_;
  SourceMap sourceMap_;
};

} // namespace torch::jit

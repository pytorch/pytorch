#pragma once

#ifdef USE_C10D_GLOO

#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <ATen/ThreadLocalState.h>

#include <torch/csrc/distributed/c10d/Work.hpp>

namespace gloo {
class Context;
} // namespace gloo

namespace c10d {

class TORCH_API ProcessGroupGlooAsyncWork : public Work {
 public:
  explicit ProcessGroupGlooAsyncWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<std::vector<at::Tensor>> outputTensors,
      OpType opType,
      uint64_t seq,
      std::chrono::milliseconds timeout,
      const char* profilingTitle = nullptr,
      const std::optional<std::vector<at::Tensor>>& inputTensors = std::nullopt);

  ~ProcessGroupGlooAsyncWork() override = default;

  static void execute(const c10::intrusive_ptr<ProcessGroupGlooAsyncWork>& work);

  virtual void run() = 0;

  std::vector<at::Tensor> result() override;

  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;
  uint64_t getSequencenumber() const override;
  std::chrono::milliseconds getTimeout() const override;
  virtual const std::vector<at::Tensor> getInputTensors() = 0;
  virtual const std::vector<at::Tensor> getOutputTensors() = 0;
  inline std::string getProfilerTitle() const {
    return profilingTitle_;
  }
  inline at::ThreadLocalState getTLS() const {
    return tls_;
  }

 protected:
  friend class ProcessGroupGloo;
  std::optional<uint64_t> trace_id_;
  std::optional<uint64_t> trace_reset_epoch_;
  std::shared_ptr<gloo::Context> context_;
  const std::chrono::milliseconds timeout_;

 private:
  void finishWorkGloo();
  void finishWorkGlooError(const std::exception_ptr& eptr);
  inline void recordAsyncWorkProfilingInfo(
      const char* profilingTitle,
      const std::optional<std::vector<at::Tensor>>& inputTensors);

  const std::vector<std::vector<at::Tensor>> outputTensors_;
  c10::intrusive_ptr<at::ivalue::Future> future_;
  std::function<void()> recordFunctionBeforeCallback_;
  const uint64_t seq_;
  std::string profilingTitle_;
  at::ThreadLocalState tls_;
};

} // namespace c10d

#endif // USE_C10D_GLOO

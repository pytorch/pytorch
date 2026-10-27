#pragma once

#include <torch/csrc/distributed/c10d/Backend.hpp>

namespace c10d {

class FakeWork : public Work {
 public:
  int seq_id = -1;

  // `result` is the tensors the collective produced. Real backends resolve
  // their future to those tensors; callers using async_op=True read them via
  // get_future().value().
  explicit FakeWork(std::vector<at::Tensor> result = {})
      : result_(std::move(result)) {}

  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
    return true;
  }

  // A fake collective is done the moment it is "issued". Work's default reads
  // completed_, which nothing here ever sets, so without this a fake op would
  // report as permanently in flight to anyone polling it even though wait()
  // returns immediately.
  bool isCompleted() override {
    return true;
  }

  std::vector<at::Tensor> result() override {
    TORCH_CHECK(!result_.empty(), "FakeWork: this work recorded no output.");
    return result_;
  }

  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
    if (result_.empty()) {
      auto fut = c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());
      fut->markCompleted();
      return fut;
    }
    auto fut = c10::make_intrusive<c10::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()));
    fut->markCompleted(result_);
    return fut;
  }

 private:
  std::vector<at::Tensor> result_;
};

// Options for the Python FakeProcessGroup
// (torch/testing/_internal/distributed/fake_pg.py). Kept in C++ so that
// ProcessGroup::splitGroup's clone() preserves the fake-specific fields.
struct FakeProcessGroupOptions : Backend::Options {
  explicit FakeProcessGroupOptions() : Backend::Options("fake") {}

  c10::intrusive_ptr<Backend::Options> clone() const override {
    return c10::make_intrusive<FakeProcessGroupOptions>(*this);
  }

  int fake_option = 0;
  bool error_on_collective = false;
  // See NOTE [FakeProcessGroup uniform-rank simulation]
  bool simulate_uniform_ranks = false;
};

} // namespace c10d

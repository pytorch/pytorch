#pragma once

#include <torch/csrc/jit/frontend/tree.h>
#include <mutex>

namespace torch::jit {

struct Call {
  std::string fn_name;
  SourceRange caller_range;
};

struct TORCH_API ErrorReport : public std::exception {
  ErrorReport(const ErrorReport& e);

  explicit ErrorReport(const SourceRange& r);
  explicit ErrorReport(const TreeRef& tree) : ErrorReport(tree->range()) {}
  explicit ErrorReport(const Token& tok) : ErrorReport(tok.range) {}

  const char* what() const noexcept override;

  class TORCH_API Calls {
   private:
    std::vector<Call> calls_;
    mutable std::mutex mutex_;

   public:
    void push_back(Call call) {
      std::lock_guard<std::mutex> lock(mutex_);
      calls_.push_back(std::move(call));
    }

    void pop_back() {
      std::lock_guard<std::mutex> lock(mutex_);
      calls_.pop_back();
    }

    bool empty() const {
      std::lock_guard<std::mutex> lock(mutex_);
      return calls_.empty();
    }

    void update_pending_range(const SourceRange& range) {
      std::lock_guard<std::mutex> lock(mutex_);
      calls_.back().caller_range = range;
    }

    std::vector<Call> get_stack() const {
      std::lock_guard<std::mutex> lock(mutex_);
      return calls_;
    }
  };

  struct TORCH_API CallStack {
    // These functions are used to report why a function was being compiled
    // (i.e. what was the call stack of user functions at compilation time that
    // led to this error)
    CallStack(const std::string& name, const SourceRange& range);
    ~CallStack();

    // Change the range that is relevant for the current function (i.e. after
    // each successful expression compilation, change it to the next expression)
    static void update_pending_range(const SourceRange& range);

   private:
    std::shared_ptr<Calls> source_callstack_;
  };

  static std::string current_call_stack();

 private:
  template <typename T>
  friend const ErrorReport& operator<<(const ErrorReport& e, const T& t);

  mutable std::stringstream ss;
  OwnedSourceRange context;
  mutable std::string the_message;
  std::vector<Call> error_stack;
};

template <typename T>
const ErrorReport& operator<<(const ErrorReport& e, const T& t) {
  e.ss << t;
  return e;
}

} // namespace torch::jit

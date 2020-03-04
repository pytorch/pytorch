#pragma once

#include "torch/csrc/WindowsTorchApiMacro.h"

#include <string>
#include <unordered_map>

namespace torch {
namespace jit {
namespace tensorexpr {

/*
ExecutionTrigger and ExecutionCounter builds instrumentation counters so
underlying functionalities can be checked.

In the code to be instrumented:

// worker.cpp
DEFINE_TRIGGER(useful_work_done);  // this defines a trigger "useful_work_done"
void run() {
  USE_TRIGGER(useful_work_done);   // this triggers the underlying counter
                                   // in  "useful_work_done"
}

// in C++ client.cpp

DECLARE_TRIGGER(useful_work_done);  // Optional: this declares a trigger that
                                    // will be defined elsewhere
ExecutionCounter counter(useful_work_done); // This starts the counter from the
                                            // underlying trigger.
... call run() ...
counter.elapsed_value();   // this returns the incremented value from the
                           // trigger since the creation of the counter

// in Python client.py
counter = ExecutionCounter("useful_work_done") // this starts the counter from
                                               // the underlying trigger
... call C++ run() ...
counter.elapsed_value()    // This returns the incremented value from the
                           // trigger since the creation of the counter.
*/

class ExecutionTrigger;
class ExecutionTriggerList {
 public:
  TORCH_API static ExecutionTriggerList& GetInstance() {
    static ExecutionTriggerList instance;
    return instance;
  }

  ExecutionTrigger* FindByName(const std::string& name) const {
    auto iter = trigger_list_.find(name);
    if (iter == trigger_list_.end()) {
      throw std::runtime_error("Invalid trigger name: " + name);
    }
    return iter->second;
  }

 private:
  friend class ExecutionTrigger;

  ExecutionTriggerList() {}
  ExecutionTriggerList(const ExecutionTriggerList&) = delete;
  ExecutionTriggerList& operator=(const ExecutionTriggerList&) = delete;

  void AddTrigger(const std::string& name, ExecutionTrigger* trigger) {
    auto insert_ret = trigger_list_.insert(std::make_pair(name, trigger));
    if (!insert_ret.second) {
      throw std::runtime_error("Duplicated trigger name: " + name);
    }
  }

  std::unordered_map<std::string, ExecutionTrigger*> trigger_list_;
};

class ExecutionTrigger {
 public:
  explicit ExecutionTrigger(const std::string& name) : name_(name) {
    ExecutionTriggerList::GetInstance().AddTrigger(name, this);
  }

  int value() const {
    return value_;
  }

  void trigger() {
    value_++;
  }

 private:
  ExecutionTrigger(const ExecutionTrigger&) = delete;
  ExecutionTrigger& operator=(const ExecutionTrigger&) = delete;
  int value_ = 0;
  const std::string name_;
};

class ExecutionCounter {
 public:
  explicit ExecutionCounter(ExecutionTrigger& trigger) : trigger_(trigger) {
    start_value_ = trigger_.value();
  }

  int elapsed_value() const {
    return trigger_.value() - start_value_;
  }

 private:
  ExecutionTrigger& trigger_;
  int start_value_ = 0;
};

#define DEFINE_TRIGGER(name) TORCH_API ExecutionTrigger name(#name)
#define DECLARE_TRIGGER(name) TORCH_API extern ExecutionTrigger name
#define USE_TRIGGER(name) (name).trigger()

} // namespace tensorexpr
} // namespace jit
} // namespace torch

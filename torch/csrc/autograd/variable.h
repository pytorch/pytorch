#pragma once

#include <mutex>
#include <memory>
#include <functional>
#include <THPP/THPP.h>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/saved_variable.h"
#include "torch/csrc/Types.h"

namespace torch { namespace autograd {

struct VariableVersion;

struct Variable : std::enable_shared_from_this<Variable> {
  Variable(
      std::unique_ptr<thpp::Tensor> data,
      std::shared_ptr<Function> grad_fn);

  Variable(
      std::unique_ptr<thpp::Tensor> data,
      bool requires_grad,
      bool is_volatile);

  SavedVariable save() const;
  static SavedVariable save_opt(Variable* var);

  std::shared_ptr<Function> get_grad_accumulator();

  static inline std::shared_ptr<Variable> of(std::unique_ptr<thpp::Tensor> data, bool is_volatile=false) {
    if (!data) {
      return std::shared_ptr<Variable>();
    }
    return std::make_shared<Variable>(std::move(data), false, is_volatile);
  }

  std::unique_ptr<thpp::Tensor> data;
  std::shared_ptr<Function> grad_fn;
  std::shared_ptr<Variable> grad;
  std::unique_ptr<VariableVersion> version_counter;
  std::vector<std::shared_ptr<FunctionPreHook>> hooks;
  std::weak_ptr<Function> grad_accumulator;
  std::mutex grad_accumulator_lock;
  bool requires_grad;
  bool is_volatile;
  int output_nr;
  PyObject *pyobj;  // weak reference
};

struct VariableVersion {
  VariableVersion() {
    saved_ref = false;
    version_block = new int[3];
    version_block[0] = 0; // version
    version_block[1] = 1; // refcount
    version_block[2] = 1; // number of variables currently using the counter
  };

  int operator++(int) { return version_block[0]++; }

  int operator*() { return *version_block; }

  int var_refcnt() { return version_block[2]; }

  void join_with(VariableVersion &other) {
    cleanup();
    version_block = other.version_block;
    version_block[1]++;
    version_block[2]++;
  }

  VariableVersion* new_saved_ref() {
    auto new_ver = new VariableVersion();
    new_ver->cleanup();
    new_ver->version_block = version_block;
    version_block[1]++;
    new_ver->saved_ref = true;
    return new_ver;
  }

  void cleanup() {
    if (!saved_ref) --version_block[2];
    if (--version_block[1]) return;
    delete[] version_block;
    version_block = nullptr;
  }

  ~VariableVersion() { cleanup(); }

  int *version_block;
  bool saved_ref;
};

}} // namespace torch::autograd

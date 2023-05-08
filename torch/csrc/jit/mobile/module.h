#pragma once
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/mobile/debug_info.h>
#include <torch/csrc/jit/mobile/function.h>
#include <torch/csrc/jit/mobile/method.h>
#include <torch/csrc/jit/mobile/quantization.h>

namespace torch {
namespace jit {
namespace mobile {
using Stack = std::vector<c10::IValue>;

// A CompilationUnit object is the one that gets executed by the lite
// interpreter.
//
// A CompilationUnit object contains a list of Method Objects. These are methods
// that appear in the original PyTorch Model. These method correspond to Python
// member functions of the Model class.
//
// Methods in turn contain a Function, and a back-pointer to the Module that
// owns this Method instance.
//
// A Function contains a Code Object (code_) which is defined in interpreter.h
//
// A Code object contains the following:
//
// std::vector<Instruction> instructions_;
// std::vector<c10::OperatorName> op_names_;
// std::vector<std::function<void(Stack&)>> operators_;
// std::vector<c10::IValue> constants_;
// std::vector<c10::TypePtr> types_;
// size_t register_size_; // Aggregated output size.
//
class CompilationUnit {
 public:
  void register_function(std::unique_ptr<Function> fn);
  std::vector<std::unique_ptr<Function>>& methods() {
    return methods_;
  }
  const std::vector<std::unique_ptr<Function>>& methods() const {
    return methods_;
  }
  Function* find_function(const c10::QualifiedName& qn);
  const Function* find_function(const c10::QualifiedName& qn) const;

  void unsafeRemoveFunction(const int64_t index) {
    methods_.erase(methods_.begin() + index);
  }

 private:
  std::vector<std::unique_ptr<Function>> methods_;
};

// A Torch Mobile Module is a representation of the model (trained in case
// of inference). A Mobile Module contains
//
// 1. data (object_)
// 2. metadata (optional) about the model (metadata_ from the metadata.pkl
//    file added after training)
// 3. Compilation Unit (cu_)
//
class TORCH_API Module {
 public:
  Module(
      c10::intrusive_ptr<c10::ivalue::Object> object,
      std::shared_ptr<CompilationUnit> cu)
      : object_(std::move(object)), cu_(std::move(cu)) {}
  Module() = default;
  Method get_method(const std::string& method_name) const;
  template <typename... Types>
  c10::IValue run_method(const std::string& method_name, Types&&... args) {
    return get_method(method_name)({IValue(std::forward<Types>(args))...});
  }
  c10::IValue forward(std::vector<c10::IValue> inputs) {
    return get_method("forward")(std::move(inputs));
  }
  c10::optional<Method> find_method(const std::string& basename) const;

  const std::string name() const {
    return object_->name();
  }
  const std::vector<at::IValue>& slots() const {
    return object_->slots();
  }
  const c10::intrusive_ptr<c10::ivalue::Object> _ivalue() const {
    return object_;
  }
  const std::vector<at::Tensor> parameters() const;
  const std::map<std::string, at::Tensor> named_parameters() const;
  std::string get_forward_method_debug_info(int64_t debug_handle) const;
  std::string getModuleHierarchy(const int64_t debug_handle) const;
  std::string getCallStack(const int64_t debug_handle) const;
  /// Enables "training" mode.
  void train(bool on = true);
  /// Calls train(false) to enable "eval" mode.
  void eval() {
    train(/*on=*/false);
  }
  /// True if the module is in training mode.
  bool is_training() const;
  const std::unordered_map<std::string, std::string> getMetadata() const {
    return metadata_;
  }
  void setMetadata(
      const std::unordered_map<std::string, std::string>& metadata) {
    metadata_ = metadata;
  }
  const std::vector<Method> get_methods() const;

  c10::IValue attr(const std::string& name, c10::IValue or_else) const {
    if (auto r = object_->type()->findAttributeSlot(name)) {
      return object_->getSlot(*r);
    }
    if (auto r = object_->type()->findConstantSlot(name)) {
      return object_->type()->getConstant(*r);
    }
    return or_else;
  }

  void setDebugTable(MobileDebugTable&& debug_table) {
    debug_table_ = std::move(debug_table);
  }
  const MobileDebugTable& getDebugTable() const {
    return debug_table_;
  }

  void setHasDebugHandles(bool has_debug_handles) {
    has_debug_handles_ = has_debug_handles;
  }

  bool hasDebugHandles() const {
    return has_debug_handles_;
  }

  const CompilationUnit& compilation_unit() const {
    return *cu_.get();
  }

  void set_delete_memory(std::shared_ptr<char> delete_mem) {
    mem_to_delete_ = delete_mem;
  }

  void set_min_operator_version(int64_t version) {
    min_operator_version_ = version;
  }

  int64_t min_operator_version() const {
    return min_operator_version_;
  }

  void set_bytecode_version(int64_t version) {
    bytecode_version_ = version;
  }

  int64_t bytecode_version() const {
    return bytecode_version_;
  }

 private:
  friend class quantization::PTQQuanizationHelper;

  bool compareMethodSchemas(
      const std::string& name_1,
      const std::string& name_2);

  void unsafeRemoveMethod(const std::string& basename);

  void unsafeCopyMethod(
      const std::string& new_method_name,
      const Function& to_be_copied);

  c10::intrusive_ptr<c10::ivalue::Object> object_;
  std::unordered_map<std::string, std::string> metadata_;
  std::shared_ptr<CompilationUnit> cu_;
  MobileDebugTable debug_table_;
  bool has_debug_handles_ = false;
  int64_t min_operator_version_ = 4;
  int64_t bytecode_version_ = 4;

  // Extra handle for the module to delete when itself is deleted
  std::shared_ptr<char> mem_to_delete_;
};

struct TORCH_API ModuleInfo {
  uint64_t bytecode_version;
  uint64_t operator_version;
  std::unordered_map<std::string, int> opname_to_num_args;
  std::unordered_set<std::string> function_names;
  std::unordered_set<std::string> type_names;
};
TORCH_API ModuleInfo get_module_info(const mobile::Module& module);

} // namespace mobile
} // namespace jit
} // namespace torch

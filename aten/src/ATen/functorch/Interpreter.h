#pragma once

#include <ATen/functorch/Macros.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/Optional.h>
#include <bitset>
#include <variant>

namespace at::functorch {

// NOTE: [functorch interpreter stack]
//
// functorch's dispatching system uses a stack of interpreters.
// Historically we've referred to this as the "DynamicLayerStack".
//
// An interpreter is something that reads in the code it is passed
// and then executes it. We have a different interpreter per-transform:
// the "VmapInterpreter" is responsible for reading in operators (like aten::mv)
// and executing the batched version of it (the batching rule for aten::mv).
//
// Concretely, each interpreter is responsible for two things:
//
// 1) process(ophandle, stack)
// Given an operator handle and a stack of arguments, the interpreter is
// responsible for figuring out how to execute the operation under the semantics
// of the interpreter. For e.g. VmapInterpreter, this is figuring out how to call
// the batching rule.
//
// The batching rules are stored as kernels on the FuncTorchBatched key, so the way
// VmapInterpreter calls the batching rule is roughly: (A) exclude all
// dispatch keys aside from the Batched key, (B) redispatch so we get to the
// Batched key.
//
// 2) sendToNextInterpreter(ophandle, stack)
// The VmapInterpreter, when it sees aten::mv, will process it into a call to
// aten::mm. It then needs to send the call to aten::mm to the next interpreter
// in the interpreter stack.
//
// The VmapInterpreter just does this via a call to ophandle.callBoxed(stack)
// and most Interpreters will implement it this way.

enum class RandomnessType {
    Error,      // always errors when calling a random function
    Same,       // randomness appears the same across batches
    Different,  // randomness appears different across batches
    END
};

enum class TransformType {
  Torch,  // Unused
  Vmap,
  Grad,  // reverse-mode AD, aka vjp
  Jvp,  // forward-mode AD
  Functionalize,
};

std::ostream& operator<<(std::ostream& os, const TransformType& t);

// NOTE: [Interpreter "subclassing" design]
//
// How are various Interpreters for different transforms (vmap, grad, ...)
// implemented?
//
// Accessing interpreters is in the hot-path of functorch so we have a constraint
// that this code must be as fast as possible.
//
// As a result, we stay away from virtual methods and this causes our code
// to look a little funny.
//
// `Interpreter` is the struct for Interpreters. It holds ALL of the
// relevant information (what type of interpreter it is and the metadata).
// Metadata for each interpreter is represented as a Union (std::variant)
// of all possible metadata (VmapInterpreterMeta, GradInterpreterMeta, ...).
//
// Given an Interpreter, how do I get a "VmapInterpreter"? You may wish to do this
// if you want to access the metadata fields (like batchSize and randomness).
//
// Each type of interpreter (e.g. Vmap) has a convenience struct
// (e.g. VmapInterpreterPtr) associated with it.
//
// Construct the convenience struct with VmapInterpreterPtr(Interpreter*),
// and then one can access methods on VmapInterpreterPtr like so:
// >>> VmapInterpreterPtr(&interpreter).batchSize()
//
// Finally, Interpreter::process switches on the type of the interpreter
// and calls one of {Transform}Intepreter::processImpl under the hood.
// Same for Interpreter::sendToNextInterpreter :)

struct VmapInterpreterMeta {
  explicit VmapInterpreterMeta(c10::SymInt batchSize, RandomnessType randomness) :
    batchSize_(std::move(batchSize)), randomness_(randomness) {}
  c10::SymInt batchSize_;
  RandomnessType randomness_;
};

struct GradInterpreterMeta {
  explicit GradInterpreterMeta(bool prevGradMode): prevGradMode_(prevGradMode) {}
  bool prevGradMode_;
};

struct JvpInterpreterMeta {
  explicit JvpInterpreterMeta(bool prevFwdGradMode) : prevFwdGradMode_(prevFwdGradMode) {}
  bool prevFwdGradMode_;
};

struct FunctionalizeInterpreterMeta {
  explicit FunctionalizeInterpreterMeta(bool functionalizeAddBackViews) :
    functionalizeAddBackViews_(functionalizeAddBackViews) {}
  bool functionalizeAddBackViews_;
};

typedef std::variant<
  int64_t,
  GradInterpreterMeta,
  JvpInterpreterMeta,
  VmapInterpreterMeta,
  FunctionalizeInterpreterMeta
> InterpreterMeta;


struct Interpreter {
  // factory functions
  static Interpreter Vmap(int64_t level, c10::SymInt batchSize, RandomnessType randomness) {
    return Interpreter(TransformType::Vmap, level, VmapInterpreterMeta(std::move(batchSize), randomness));
  }
  static Interpreter Grad(int64_t level, bool prevGradMode) {
    return Interpreter(TransformType::Grad, level, GradInterpreterMeta(prevGradMode));
  }
  static Interpreter Jvp(int64_t level, bool prevFwdGradMode) {
    return Interpreter(TransformType::Jvp, level, JvpInterpreterMeta(prevFwdGradMode));
  }
  static Interpreter Functionalize(int64_t level, bool functionalizeAddBackViews) {
    return Interpreter(TransformType::Functionalize, level, FunctionalizeInterpreterMeta(functionalizeAddBackViews));
  }

  // methods
  TransformType key() const { return type_; }
  int64_t level() const { return level_; }
  const InterpreterMeta& meta() const { return meta_; }

  void process(const c10::OperatorHandle& op, torch::jit::Stack* stack);
  void sendToNextInterpreter(const c10::OperatorHandle& op, torch::jit::Stack* stack, bool grad_special_case);

  void saveLocalDispatchKeySet(c10::impl::LocalDispatchKeySet keyset) {
    TORCH_INTERNAL_ASSERT(!savedLocalDispatchKeySet_.has_value());
    savedLocalDispatchKeySet_ = std::move(keyset);
  }
  void clearSavedLocalDispatchKeySet() {
    TORCH_INTERNAL_ASSERT(savedLocalDispatchKeySet_.has_value());
    savedLocalDispatchKeySet_ = c10::nullopt;
  }
  c10::impl::LocalDispatchKeySet getSavedLocalDispatchKeySet() const {
    TORCH_INTERNAL_ASSERT(savedLocalDispatchKeySet_.has_value());
    return *savedLocalDispatchKeySet_;
  }

  // An Interpreter is alive if we are currently inside the ongoing transform
  // for the interpreter. For example, vmap(f)(x); inside of f, the vmap's
  // corresponding Interpreter is alive, even when it is not on the DynamicLayerStack.
  bool is_alive() const {
    return *is_alive_;
  }
  const std::shared_ptr<bool>& is_alive_ptr() const {
    return is_alive_;
  }
  void set_is_alive(bool alive) {
    *is_alive_ = alive;
  }

  // Please don't use this
  explicit Interpreter() = default;

 private:
  explicit Interpreter(TransformType type, int64_t level, InterpreterMeta meta):
    type_(type), level_(level), is_alive_(std::make_shared<bool>(false)), meta_(meta) {}

  // fields
  TransformType type_;
  int64_t level_;
  optional<c10::impl::LocalDispatchKeySet> savedLocalDispatchKeySet_;
  std::shared_ptr<bool> is_alive_;
  InterpreterMeta meta_;
};

// Applies the following for-loop:
// for i in range(begin, end):
//   args[i] = func(args[i])
void foreachTensorInplace(std::vector<IValue>& args, int64_t begin, int64_t end,
    std::function<Tensor(const Tensor&)> func);

// Applies the following for-loop:
// for i in range(begin, end):
//   if use_flag_relative[i] == 1: <-- treats use_flag_relative as a bitset
//     args[i] = func(args[i], i - begin, true)
//   args[i] = func(args[i], i - begin)
void foreachTensorInplaceWithFlag(std::vector<IValue>& args, int64_t begin, int64_t end,
    const std::bitset<64> use_flag_relative, std::function<Tensor(const Tensor&, bool)> func);

std::vector<int64_t> findUnwrappedInputs(std::vector<IValue>& args, int64_t begin, int64_t end);

DispatchKeySet keysToExcludeWhenEnteringDynamicLayer(TransformType key);

void setup_dispatch_key_tls(TransformType key, DispatchKeySet include);

void sanityCheckStack(const c10::OperatorHandle& op, torch::jit::Stack* stack);

} // namespace at::functorch

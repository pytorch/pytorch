#pragma once

#include <ATen/functorch/Macros.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <optional>
#include <bitset>
#include <utility>
#include <variant>

#include <nlohmann/json.hpp>

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

  VmapInterpreterMeta() = default;
  VmapInterpreterMeta(const VmapInterpreterMeta&) = default;
  VmapInterpreterMeta(VmapInterpreterMeta&&) = default;
  VmapInterpreterMeta& operator=(const VmapInterpreterMeta&) = default;
  VmapInterpreterMeta& operator=(VmapInterpreterMeta&&) = default;
  ~VmapInterpreterMeta() = default;

  template <typename T>
  friend void to_json(T& json_j, const VmapInterpreterMeta& json_t) {
    if (json_t.batchSize_.is_heap_allocated()) {
      throw std::runtime_error("Serialization for heap-allocated SymInt is not implemented yet");
    }
    json_j["batchSize"] = json_t.batchSize_.as_int_unchecked();
    json_j["randomness"] = static_cast<int64_t>(json_t.randomness_);
  }

  template <typename T>
  friend void from_json(const T& json_j, VmapInterpreterMeta& json_t) {
    json_t.batchSize_ = c10::SymInt(SymInt::Unchecked::UNCHECKED, json_j["batchSize"]);
    json_t.randomness_ = static_cast<RandomnessType>(json_j["randomness"]);
  }
};

struct GradInterpreterMeta {
  explicit GradInterpreterMeta(bool prevGradMode): prevGradMode_(prevGradMode) {}
  GradInterpreterMeta() = default;
  GradInterpreterMeta(const GradInterpreterMeta&) = default;
  GradInterpreterMeta(GradInterpreterMeta&&) = default;
  GradInterpreterMeta& operator=(const GradInterpreterMeta&) = default;
  GradInterpreterMeta& operator=(GradInterpreterMeta&&) = default;
  ~GradInterpreterMeta() = default;

  bool prevGradMode_;
  template <typename T>
  friend void to_json(T& json_j, const GradInterpreterMeta& json_t) {
    json_j["prevGradMode"] = json_t.prevGradMode_;
  }

  template <typename T>
  friend void from_json(const T& json_j, GradInterpreterMeta& json_t) {
    json_t.prevGradMode_ = json_j["prevGradMode"];
  }
};

struct JvpInterpreterMeta {
  explicit JvpInterpreterMeta(bool prevFwdGradMode) : prevFwdGradMode_(prevFwdGradMode) {}
  JvpInterpreterMeta() = default;
  JvpInterpreterMeta(const JvpInterpreterMeta&) = default;
  JvpInterpreterMeta(JvpInterpreterMeta&&) = default;
  JvpInterpreterMeta& operator=(const JvpInterpreterMeta&) = default;
  JvpInterpreterMeta& operator=(JvpInterpreterMeta&&) = default;
  ~JvpInterpreterMeta() = default;

  bool prevFwdGradMode_;
  template <typename T>
  friend void to_json(T& json_j, const JvpInterpreterMeta& json_t) {
    json_j["prevFwdGradMode"] = json_t.prevFwdGradMode_;
  }

  template <typename T>
  friend void from_json(const T& json_j, JvpInterpreterMeta& json_t) {
    json_t.prevFwdGradMode_ = json_j["prevFwdGradMode"];
  }
};

struct FunctionalizeInterpreterMeta {
  explicit FunctionalizeInterpreterMeta(bool functionalizeAddBackViews) :
    functionalizeAddBackViews_(functionalizeAddBackViews) {}
  FunctionalizeInterpreterMeta() = default;
  FunctionalizeInterpreterMeta(const FunctionalizeInterpreterMeta&) = default;
  FunctionalizeInterpreterMeta(FunctionalizeInterpreterMeta&&) = default;
  FunctionalizeInterpreterMeta& operator=(const FunctionalizeInterpreterMeta&) = default;
  FunctionalizeInterpreterMeta& operator=(FunctionalizeInterpreterMeta&&) = default;
  ~FunctionalizeInterpreterMeta() = default;

  bool functionalizeAddBackViews_;
  template <typename T>
  friend void to_json(T& json_j, const FunctionalizeInterpreterMeta& json_t) {
    json_j["functionalizeAddBackViews"] = json_t.functionalizeAddBackViews_;
  }

  template <typename T>
  friend void from_json(const T& json_j, FunctionalizeInterpreterMeta& json_t) {
    json_t.functionalizeAddBackViews_ = json_j["functionalizeAddBackViews"];
  }
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
    savedLocalDispatchKeySet_ = keyset;
  }
  void clearSavedLocalDispatchKeySet() {
    TORCH_INTERNAL_ASSERT(savedLocalDispatchKeySet_.has_value());
    savedLocalDispatchKeySet_ = std::nullopt;
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

  template <typename T>
  friend void to_json(T& json_j, const Interpreter& json_t) {
    json_j["type"] = static_cast<int64_t>(json_t.type_);
    json_j["level"] = json_t.level_;
    if (json_t.savedLocalDispatchKeySet_) {
      json_j["savedLocalDispatchKeySet"] = {
        {"included", json_t.savedLocalDispatchKeySet_->included_.raw_repr()},
        {"excluded", json_t.savedLocalDispatchKeySet_->excluded_.raw_repr()}
      };
    } else {
      json_j["savedLocalDispatchKeySet"] = nlohmann::json();
    }
    json_j["is_alive"] = *json_t.is_alive_;
    std::visit([&](auto&& arg) {
        using V = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<V, int64_t>) {
          json_j["meta"] = {{"Torch", arg}};
        } else if constexpr (std::is_same_v<V, GradInterpreterMeta>) {
          json_j["meta"] = {{"Grad", arg}};
        } else if constexpr (std::is_same_v<V, JvpInterpreterMeta>) {
          json_j["meta"] = {{"Jvp", arg}};
        } else if constexpr (std::is_same_v<V, VmapInterpreterMeta>) {
          json_j["meta"] = {{"Vmap", arg}};
        } else if constexpr (std::is_same_v<V, FunctionalizeInterpreterMeta>) {
          json_j["meta"] = {{"Functionalize", arg}};
        } else {
          static_assert(false && sizeof(V), "unknown variant case");
        }
    }, json_t.meta_);
  }

  template <typename T>
  friend void from_json(const T& json_j, Interpreter& json_t) {
    json_t.type_ = static_cast<TransformType>(json_j["type"]);
    json_t.level_ = json_j["level"];
    auto savedLocalDispatchKeySet = json_j["savedLocalDispatchKeySet"];
    if (savedLocalDispatchKeySet.is_null()) {
      json_t.savedLocalDispatchKeySet_ = std::nullopt;
    } else {
      c10::impl::PODLocalDispatchKeySet pod;
      pod.set_included(DispatchKeySet::from_raw_repr(savedLocalDispatchKeySet["included"].template get<uint64_t>()));
      pod.set_excluded(DispatchKeySet::from_raw_repr(savedLocalDispatchKeySet["excluded"].template get<uint64_t>()));
      json_t.savedLocalDispatchKeySet_ = c10::impl::LocalDispatchKeySet(pod);
    }
    json_t.is_alive_ = std::make_shared<bool>(json_j["is_alive"]);
    auto meta = json_j["meta"];
    if (meta.contains("Torch")) {
      json_t.meta_.emplace<int64_t>(meta["Torch"].template get<int64_t>());
    } else if (meta.contains("Grad")) {
      json_t.meta_.emplace<GradInterpreterMeta>(meta["Grad"].template get<GradInterpreterMeta>());
    } else if (meta.contains("Jvp")) {
      json_t.meta_.emplace<JvpInterpreterMeta>(meta["Jvp"].template get<JvpInterpreterMeta>());
    } else if (meta.contains("Vmap")) {
      json_t.meta_.emplace<VmapInterpreterMeta>(meta["Vmap"].template get<VmapInterpreterMeta>());
    } else if (meta.contains("Functionalize")) {
      json_t.meta_.emplace<FunctionalizeInterpreterMeta>(meta["Functionalize"].template get<FunctionalizeInterpreterMeta>());
    } else {
      throw std::runtime_error("unknown interpreter metadata type");
    }
  }

  std::string serialize() const {
    return nlohmann::json(*this).dump();
  }

  static Interpreter deserialize(const std::string& serialized) {
    return nlohmann::json::parse(serialized).get<Interpreter>();
  }

 private:
  explicit Interpreter(TransformType type, int64_t level, InterpreterMeta meta):
    type_(type), level_(level), is_alive_(std::make_shared<bool>(false)), meta_(std::move(meta)) {}

  // fields
  TransformType type_{};
  int64_t level_{};
  std::optional<c10::impl::LocalDispatchKeySet> savedLocalDispatchKeySet_;
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
    const std::bitset<64> use_flag_relative, const std::function<Tensor(const Tensor&, bool)>& func);

std::vector<int64_t> findUnwrappedInputs(std::vector<IValue>& args, int64_t begin, int64_t end);

DispatchKeySet keysToExcludeWhenEnteringDynamicLayer(TransformType key);

void setup_dispatch_key_tls(TransformType key, DispatchKeySet include);

void sanityCheckStack(const c10::OperatorHandle& op, torch::jit::Stack* stack);

} // namespace at::functorch

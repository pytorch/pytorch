#pragma once

#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/variable_tensor_list.h>
#include <torch/csrc/utils/hash.h>
#include <iostream>
#include <vector>

namespace torch {
namespace jit {

// GraphExecutor creates specializations of Graphs for different
// dimensionalitities and types of inputs.

inline static at::Device ConvertIntToCPUOrCUDA(int device) {
  return device < 0 ? at::kCPU : at::Device(at::DeviceType::CUDA, device);
}
struct ArgumentInfo {
  friend struct ArgumentSpec;
  using plain_data_type = uint32_t;

  bool defined() const {
    return defined_;
  }
  bool isNone() const {
    return is_none_;
  }
  bool isNontensorOptional() const {
    return is_nontensor_optional_;
  }
  int device() const {
    return device_;
  }
  // XXX: It is guaranteed that this will return false when called on non-tensor
  // arguments
  bool requires_grad() const {
    return requires_grad_;
  }
  int dim() const {
    return dim_;
  }
  at::ScalarType type() const {
    return at::ScalarType(type_);
  }
  operator TypePtr() const {
    if (!defined())
      return TensorType::get();
    return DimensionedTensorType::create(
        type(), ConvertIntToCPUOrCUDA(device()), dim());
  }

 private:
  unsigned is_nontensor_optional_ : 1;
    // =0 if it is an Optional[Tensor] or Tensor. =1 other optional type.
    // if this is set, values below is_none_ must be 0
    // Non-optional other types are not considered
  unsigned is_none_ : 1;
    // is_none_  =1 if an Optional[T] is None, remaining bits undefined
  unsigned defined_ : 1;
  unsigned requires_grad_ : 1;
  unsigned : 4;
  unsigned dim_ : 8;
  int device_ : 8; // NOTE: this needs to be signed because we use -1 to
                   // represent CPU
  unsigned type_ : 8;
};

static_assert(
    std::is_pod<ArgumentInfo>::value,
    "ArgumentInfo is to be a POD struct");
static_assert(
    sizeof(ArgumentInfo) == sizeof(ArgumentInfo::plain_data_type),
    "ArgumentInfo is expected to be a 32-bit struct");

struct ArgumentSpec {
  ArgumentSpec(size_t num_flat_inputs) {
    hash_code = num_flat_inputs;
    args.reserve(num_flat_inputs);
  }

  void addNontensorOptional(const IValue& input) {
    args.emplace_back();
    auto& arg = args.back();
    // Initialize all fields to 0. This is convenient, because e.g.
    // requires_grad() can be checked even on tensors AND will make
    // padding bits all 0s.
    std::memset(&arg, 0, sizeof(ArgumentInfo));
    arg.is_nontensor_optional_ = true;
    arg.is_none_ = input.isNone();
    combineHash(arg);
  }

  void addTensor(const IValue& input, bool with_grad) {
    args.emplace_back();
    auto& arg = args.back();
    // Initialize all fields to 0. This is convenient, because e.g.
    // requires_grad() can be checked even on tensors AND will make
    // padding bits all 0s.
    std::memset(&arg, 0, sizeof(ArgumentInfo));

    if (input.isNone()) {
      arg.is_none_ = 1;
    } else {
      AT_ASSERT(input.isTensor());
      // [argspec refcounting] reinterpret the IValue to avoid having to refcount
      // the Tensor microbenchmarks
      // https://github.com/zdevito/pytorch/commit/21e7200a0a0fc456bea2f10e95b1781f83933d10
      // show overhead in extra refcounting along this path
      const at::Tensor* t = reinterpret_cast<const at::Tensor*>(&input);
      if ((arg.defined_ = t->defined())) {
	arg.requires_grad_ = with_grad && autograd::Variable(*t).requires_grad();
	arg.dim_ = t->dim();
	arg.device_ = t->is_cuda() ? t->get_device() : -1;
	arg.type_ = static_cast<unsigned>(t->scalar_type());
      }
    }
    combineHash(arg);
  }

  void combineHash(const ArgumentInfo& arg) {
    ArgumentInfo::plain_data_type arg_data;
    std::memcpy(&arg_data, &arg, sizeof(ArgumentInfo));
    hash_code = hash_combine(hash_code, arg_data);
  }

  // equality is fast: check ninputs, and then check the raw array data,
  // there are no size/stride indirections
  bool operator==(const ArgumentSpec& spec) const {
    if (args.size() != spec.args.size())
      return false;
    // NB: we need to break out early when there are no elements, because
    // passing a nullptr to memcmp is UB.
    if (args.size() == 0)
      return true;
    return std::memcmp(
               args.data(),
               spec.args.data(),
               args.size() * sizeof(ArgumentInfo)) == 0;
  }
  bool operator!=(const ArgumentSpec& spec) const {
    return !(*this == spec);
  }
  size_t size() const {
    return args.size();
  }
  const ArgumentInfo& at(size_t i) const {
    return args[i];
  }
  size_t hashCode() const {
    return hash_code;
  }

 private:
  size_t hash_code; // precomputed on construction
  std::vector<ArgumentInfo> args;
};

// ArgumentSpecCreator takes an initial graph and comes up with a set
// of simple instructions to compute the ArgumentSpec given a set of
// input tensors.
struct TORCH_API ArgumentSpecCreator {
  // instructs acts on a stack of a list of input IValues
  // at the beginning the stack contains a single list of the inputs to the
  // function the ENTER_ instructs descend into subobjects and push new lists
  // onto the stack
  enum Inst : char {
    ENTER_TUPLE, // consume a tuple ivalue from the top-most list, and push the
                 // list of its elements onto the stack as a new list
    ENTER_OBJECT, // same as ENTER_TUPLE, but the input is a class
    LEAVE, // pop the top-most list from the stack
    SKIP, // consume an element from the top-most list, and discard
    SPECIALIZE_TENSOR, // consume a tensor or optional tensor for the top-most
                       // list, and add it to the ArgSpec key being created
    SPECIALIZE_NONTENSOR_OPTIONAL,
                       // consume a nontensor optional from the top-most list,
                       // and add it to the ArgSpec key being created
  };
  ArgumentSpecCreator(Graph& graph);
  ArgumentSpec create(bool with_grad, const Stack& stack) const;
  void setInputTypes(Graph& g, const ArgumentSpec& spec) const;
  std::vector<TypePtr> getSpecializedTypes(
      Graph& graph,
      const ArgumentSpec& spec) const;
  void dump() const;
  using WrittenSlots = std::unordered_set<std::string>;

 private:
  static constexpr size_t DEPTH_LIMIT = 128;
  void scan(
      const TypePtr& typ,
      size_t depth,
      const WrittenSlots& written_slots);
  size_t num_inputs_;
  size_t num_tensors_or_optionals_ = 0;
  std::vector<Inst> instructions_;
};

inline std::ostream& operator<<(std::ostream& out, const ArgumentInfo& info) {
  if (!info.defined()) {
    return out << "<undefined>";
  }
  out << "Tensor(device=" << info.device() << ", type=" << toString(info.type())
      << ", requires_grad=" << info.requires_grad() << ", dims=" << info.dim()
      << ")";
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const ArgumentSpec& spec) {
  out << "{";
  for (size_t i = 0; i < spec.size(); ++i) {
    if (i > 0)
      out << ", ";
    out << spec.at(i);
  }
  out << "}";
  return out;
}

} // namespace jit
} // namespace torch

namespace std {
template <>
struct hash<torch::jit::ArgumentSpec> {
  size_t operator()(const torch::jit::ArgumentSpec& spec) const {
    return spec.hashCode();
  }
};
} // namespace std

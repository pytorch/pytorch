#pragma once

#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>
#include <c10/util/hash.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/ir/ir.h>
#include <iostream>
#include <vector>

namespace torch {
namespace jit {

// GraphExecutor creates specializations of Graphs for different
// dimensionalitities and types of inputs.

inline static at::Device ConvertIntToCPUOrCUDA(int device) {
  return device < 0 ? at::kCPU : at::Device(DeviceType::CUDA, device);
}
struct ArgumentInfo {
  friend struct ArgumentSpec;
  using plain_data_type = uint32_t;

  bool defined() const {
    return defined_;
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
  TypePtr toType() const {
    if (!defined())
      return TensorType::get();

    return TensorType::create(
        type(),
        ConvertIntToCPUOrCUDA(device()),
        c10::optional<size_t>(dim()),
        requires_grad());
  }
  operator TypePtr() const {
    return toType();
  }

 private:
  unsigned defined_ : 1;
  unsigned requires_grad_ : 1;
  unsigned : 5;
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
  ArgumentSpec(size_t num_flat_tensor_inputs, size_t num_flat_optional_inputs) {
    hash_code =
        c10::hash_combine(num_flat_tensor_inputs, num_flat_optional_inputs);
    tensor_args.reserve(num_flat_tensor_inputs);
    optional_presence.reserve(num_flat_optional_inputs);
  }

  void addOptional(const IValue& input) {
    bool is_present = !input.isNone();
    optional_presence.push_back(is_present);
    hash_code = c10::hash_combine(hash_code, is_present);
  }

  void addTensor(const IValue& input, bool with_grad) {
    AT_ASSERT(input.isTensor(), "Expected Tensor but found ", input.tagKind());
    tensor_args.emplace_back();
    auto& arg = tensor_args.back();
    // Initialize all fields to 0. This is convenient, because e.g.
    // requires_grad() can be checked even on tensors AND will make
    // padding bits all 0s.
    std::memset(&arg, 0, sizeof(ArgumentInfo));

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
    combineHash(arg);
  }

  void combineHash(const ArgumentInfo& arg) {
    ArgumentInfo::plain_data_type arg_data;
    std::memcpy(&arg_data, &arg, sizeof(ArgumentInfo));
    hash_code = c10::hash_combine(hash_code, arg_data);
  }

  // equality is fast: check ninputs, and then check the raw array data,
  // there are no size/stride indirections
  // hopefully std::vector<bool> has fast equality
  bool operator==(const ArgumentSpec& spec) const {
    if (optional_presence != spec.optional_presence) {
      return false;
    }
    if (tensor_args.size() != spec.tensor_args.size())
      return false;
    // NB: we need to break out early when there are no elements, because
    // passing a nullptr to memcmp is UB.
    if (tensor_args.size() == 0)
      return true;
    return std::memcmp(
               tensor_args.data(),
               spec.tensor_args.data(),
               tensor_args.size() * sizeof(ArgumentInfo)) == 0;
  }
  bool operator!=(const ArgumentSpec& spec) const {
    return !(*this == spec);
  }
  size_t numTensors() const {
    return tensor_args.size();
  }
  const ArgumentInfo& tensorAt(size_t i) const {
    return tensor_args[i];
  }
  size_t numOptionals() const {
    return optional_presence.size();
  }
  bool isPresent(size_t i) const {
    return optional_presence[i];
  }
  size_t hashCode() const {
    return hash_code;
  }

 private:
  size_t hash_code; // precomputed on construction
  std::vector<ArgumentInfo> tensor_args;
  std::vector<bool> optional_presence;
};

namespace {
static constexpr size_t ARG_SPEC_DEPTH_LIMIT = 128;
}

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
    SPECIALIZE_OPTIONAL_TENSOR, // consume a optional tensor for the top-most
                                // list, and add it to the ArgSpec key being
                                // created
    SPECIALIZE_TENSOR, // consume a tensor for the top-most
                       // list, and add it to the ArgSpec key being created
    SPECIALIZE_OPTIONAL,
    // consume a nontensor optional from the top-most list,
    // and add it to the ArgSpec key being created
  };
  ArgumentSpecCreator(Graph& graph);
  ArgumentSpec create(bool with_grad, const Stack& stack) const;
  void specializeTypes(Graph& g, const ArgumentSpec& spec) const;
  void dump() const;
  using WrittenSlots = std::unordered_set<std::string>;

 private:
  void scan(
      const TypePtr& typ,
      size_t depth,
      const WrittenSlots& written_slots);
  size_t num_inputs_;
  size_t num_tensors_ = 0;
  size_t num_optionals_ = 0;
  std::vector<Inst> instructions_;
};

// CompleteArgumentSpec represents one particular specialization.
// It is designed so that it can be created, hashed, and compared quickly
// since it is used along the hot-path of the JIT to check if the code
// we have created is valid for the given inputs.

// COmpleteArgumentInfoPOD is only used internally in CompleteArgumentSpec
// API users should use ArgumentInfo
struct CompleteArgumentInfoPOD {
  // total size is 64-bit
  unsigned is_tensor : 8; // all other fields are invalid if this is false
  unsigned type : 8; // scalar type
  unsigned defined : 1;
  unsigned requires_grad : 1;
  signed device : 14;
  uint32_t total_dims; // all TensorInfoPODs are in CompleteArgumentSpec's
                       // tensor_info() array. total_dims is the total number of
                       // dimensions seen so far in all previous members of
                       // tensor_info(), including this tensor 2*total_dims
                       // becomes the offset into the sizes_strides list for the
                       // _next_ tensor in the tensor_info array for tensor 0,
                       // the offset is always 0
};

static_assert(
    sizeof(CompleteArgumentInfoPOD) == sizeof(int64_t),
    "CompleteArgumentInfoPOD must be 64-bit struct for CompleteArgumentSpec encoding to work");

struct CompleteArgumentInfo;

struct CompleteArgumentSpec {
  CompleteArgumentSpec(bool with_grad, at::ArrayRef<IValue> inputs)
      : hash_code(0), ninputs(inputs.size()) {
    int32_t all_dims = 0;
    const int32_t num_inputs = inputs.size();
    for (int32_t i = 0; i < num_inputs; i++) {
      if (!inputs[i].isTensor())
        continue;
      auto tensor = inputs[i].toTensor();
      all_dims += tensor.defined() ? tensor.ndimension() : 0;
    }
    // allocate enough room for all TensorPODs and dimensions
    data.resize(ninputs + all_dims * 2);

    // and reinterpret our data array as these structs
    auto* pods = reinterpret_cast<CompleteArgumentInfoPOD*>(data.data());
    int64_t* next_dim = sizes_strides();
    int32_t total_dims = 0;
    for (int32_t i = 0; i < num_inputs; i++) {
      auto& pod = pods[i];
      pod.is_tensor = static_cast<uint32_t>(inputs[i].isTensor());
      if (pod.is_tensor) {
        at::Tensor t = inputs[i].toTensor();
        pod.defined = t.defined();
        if (pod.defined) {
          pod.type = static_cast<int>(t.scalar_type());
          pod.device = (!t.is_cuda()) ? -1 : t.get_device();
          pod.requires_grad = with_grad && t.requires_grad();
          total_dims += t.ndimension();
          auto sizes = t.sizes();
          std::copy(sizes.begin(), sizes.end(), next_dim);
          next_dim += sizes.size();
          auto strides = t.strides();
          std::copy(strides.begin(), strides.end(), next_dim);
          next_dim += strides.size();
        }
      }
      // each POD has a running tally of all dimensions including its own
      pod.total_dims = total_dims;
    }
    // we precompute the hash_code to minimize the time inside of hash
    // table operations where we may need to hold a compiler cache lock.
    hash_code = c10::hash_combine(0, ninputs);
    for (auto d : data) {
      hash_code = c10::hash_combine(hash_code, d);
    }
  }

  // equality is fast: check ninputs, and then check the raw array data,
  // there are no size/stride indirections
  bool operator==(const CompleteArgumentSpec& spec) const {
    return ninputs == spec.ninputs && data == spec.data;
  }
  bool operator!=(const CompleteArgumentSpec& spec) const {
    return !(*this == spec);
  }
  friend struct CompleteArgumentInfo;
  CompleteArgumentInfo at(size_t i) const;
  size_t size() const {
    return ninputs;
  }
  size_t hashCode() const {
    return hash_code;
  }

 private:
  ArrayRef<CompleteArgumentInfoPOD> tensor_info() const {
    return ArrayRef<CompleteArgumentInfoPOD>(
        reinterpret_cast<const CompleteArgumentInfoPOD*>(data.data()), ninputs);
  }
  // the start of the sizes_strides information, which comes after the
  // CompleteArgumentInfoPOD list.
  const int64_t* sizes_strides() const {
    return data.data() + ninputs;
  }
  int64_t* sizes_strides() {
    return data.data() + ninputs;
  }
  size_t hash_code; // precomputed on construction
  size_t ninputs;
  // layout is ninputs of TensorPOD (each 64-bit) followed by their size and
  // stride info for 3 tensors:
  // [t0POD][t1POD][t2POD]...
  // [t0 sizes][t0 strides][t1 sizes][t1 strides][t2 sizes][t2 strides]
  std::vector<int64_t> data;
};

// public view of compressed CompleteArgumentInfo
struct CompleteArgumentInfo {
  CompleteArgumentInfo(const CompleteArgumentSpec& spec, const int i)
      : spec(spec), i(i) {}
  bool isTensor() const {
    return pod(i).is_tensor;
  }
  at::ScalarType type() const {
    return at::ScalarType(pod(i).type);
  }
  bool defined() const {
    return pod(i).defined;
  }
  bool requires_grad() const {
    return pod(i).requires_grad;
  }
  int device() const {
    return pod(i).device;
  }
  int ndimension() const {
    // See [valid range], it is always valid to ask for offset for (i + 1)
    return (sizes_strides_offset(i + 1) - sizes_strides_offset(i)) / 2;
  }
  at::IntArrayRef sizes() const {
    return at::IntArrayRef(
        spec.sizes_strides() + sizes_strides_offset(i), ndimension());
  }
  at::IntArrayRef strides() const {
    int ndim = ndimension();
    return at::IntArrayRef(
        spec.sizes_strides() + sizes_strides_offset(i) + ndim, ndim);
  }
  operator TypePtr() const {
    if (!defined())
      return TensorType::get();
    return TensorType::create(
        type(),
        ConvertIntToCPUOrCUDA(device()),
        c10::VaryingShape<int64_t>{sizes()},
        c10::VaryingShape<int64_t>{strides()},
        requires_grad());
  }

 private:
  // offsetinto sizes_strides() array where the sizes start for tensor j
  // [valid range] valid range is [0, ninputs]
  // (i.e. you can ask for the offset at ninputs, which would be the offset of
  // the next tensor if it existed)
  int sizes_strides_offset(int j) const {
    if (j == 0)
      return 0;
    return 2 * pod(j - 1).total_dims;
  }
  const CompleteArgumentInfoPOD& pod(int j) const {
    return spec.tensor_info().at(j);
  }
  const CompleteArgumentSpec& spec;
  const int i;
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
  for (size_t i = 0; i < spec.numTensors(); ++i) {
    if (i > 0)
      out << ", ";
    out << spec.tensorAt(i);
  }
  out << "; ";
  for (size_t i = 0; i < spec.numOptionals(); ++i) {
    if (i > 0)
      out << ", ";
    out << spec.isPresent(i);
  }
  out << "}";
  return out;
}

inline std::ostream& operator<<(
    std::ostream& out,
    const CompleteArgumentInfo& info) {
  if (!info.defined()) {
    return out << "<undefined>";
  }
  out << "Tensor(device=" << info.device() << ", type=" << toString(info.type())
      << ", requires_grad=" << info.requires_grad()
      << ", sizes=" << info.sizes() << ", strides=" << info.strides() << ")";
  return out;
}

inline std::ostream& operator<<(
    std::ostream& out,
    const CompleteArgumentSpec& spec) {
  out << "{";
  for (size_t i = 0; i < spec.size(); ++i) {
    if (i > 0)
      out << ", ";
    out << spec.at(i);
  }
  out << "}";
  return out;
}

inline CompleteArgumentInfo CompleteArgumentSpec::at(size_t i) const {
  return CompleteArgumentInfo(*this, i);
}

inline c10::optional<int8_t> convertOptional(
    c10::optional<c10::ScalarType> const& from) {
  return (from) ? c10::optional<int8_t>(static_cast<int8_t>(*from))
                : c10::optional<int8_t>{};
}

} // namespace jit
} // namespace torch

namespace std {

template <typename T>
struct hash<c10::VaryingShape<T>> {
  size_t operator()(const c10::VaryingShape<T>& vs) const {
    return c10::get_hash(
        vs.size(),
        vs.size() ? vs.sizes().value() : std::vector<c10::optional<T>>());
  }
};

template <>
struct hash<c10::TensorType> {
  size_t operator()(const c10::TensorType& ptt) const {
    return c10::get_hash<
        c10::optional<int8_t>,
        c10::VaryingShape<int64_t>,
        c10::VaryingShape<int64_t>,
        c10::optional<bool>>(
        torch::jit::convertOptional(ptt.scalarType()),
        ptt.sizes(),
        ptt.strides(),
        ptt.requiresGrad());
  }
};

template <>
struct hash<torch::jit::ArgumentSpec> {
  size_t operator()(const torch::jit::ArgumentSpec& spec) const {
    return spec.hashCode();
  }
};
template <>
struct hash<torch::jit::CompleteArgumentSpec> {
  size_t operator()(const torch::jit::CompleteArgumentSpec& spec) const {
    return spec.hashCode();
  }
};
} // namespace std

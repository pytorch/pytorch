#pragma once

#include <iostream>
#include <vector>
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/utils/hash.h"
#include "torch/csrc/jit/stack.h"
#include "torch/csrc/jit/variable_tensor_list.h"

namespace torch { namespace jit {

// GraphExecutor creates specializations of Graphs for different dimensionalitities
// and types of inputs.

// ArgumentSpec represents one particular specialization.
// It is designed so that it can be created, hashed, and compared quickly
// since it is used along the hot-path of the JIT to check if the code
// we have created is valid for the given inputs.

// ArgumentInfoPOD is only used internally in ArgumentSpec
// API users should use ArgumentInfo
struct ArgumentInfoPOD {
  // total size is 64-bit
  unsigned is_tensor : 8; // all other fields are invalid if this is false
  unsigned type : 8; // scalar type
  unsigned defined : 1;
  unsigned requires_grad : 1;
  signed device : 14;
  uint32_t total_dims; // all TensorInfoPODs are in ArgumentSpec's tensor_info() array.
                       // total_dims is the total number of dimensions seen so far
                       // in all previous members of tensor_info(), including this tensor
                       // 2*total_dims becomes the offset into the sizes_strides list
                       // for the _next_ tensor in the tensor_info array
                       // for tensor 0, the offset is always 0
};

static_assert(sizeof(ArgumentInfoPOD) == sizeof(int64_t),
  "ArgumentInfoPOD must be 64-bit struct for ArgumentSpec encoding to work");

struct ArgumentInfo;

struct ArgumentSpec {
  ArgumentSpec(bool with_grad, at::ArrayRef<IValue> inputs)
  :  hash_code(0), ninputs(inputs.size()) {
    int32_t all_dims = 0;
    const int32_t num_inputs = inputs.size();
    for (int32_t i = 0; i < num_inputs; i++) {
      if (!inputs[i].isTensor()) continue;
      auto tensor = inputs[i].toTensor();
      all_dims += tensor.defined() ? tensor.ndimension() : 0;
    }
    // allocate enough room for all TensorPODs and dimensions
    data.resize(ninputs + all_dims*2);

    // and reinterpret our data array as these structs
    ArgumentInfoPOD * pods = reinterpret_cast<ArgumentInfoPOD*>(data.data());
    int64_t * next_dim = sizes_strides();
    int32_t total_dims = 0;
    for(int32_t i = 0; i < num_inputs; i++) {
      auto & pod = pods[i];
      pod.is_tensor = static_cast<uint32_t>(inputs[i].isTensor());
      if (pod.is_tensor) {
        at::Tensor t = inputs[i].toTensor();
        pod.defined = t.defined();
        if (pod.defined) {
          pod.type = static_cast<int>(t.type().scalarType());
          pod.device = (!t.type().is_cuda()) ? -1 : t.get_device();
          pod.requires_grad = with_grad && autograd::as_variable_ref(t).requires_grad();
          total_dims += t.ndimension();
          auto sizes = t.sizes();
          std::copy(sizes.begin(),sizes.end(), next_dim);
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
    hash_code = hash_combine(0, ninputs);
    for(auto d : data) {
      hash_code = hash_combine(hash_code, d);
    }
  }

  // equality is fast: check ninputs, and then check the raw array data,
  // there are no size/stride indirections
  bool operator==(const ArgumentSpec & spec) const {
    return ninputs == spec.ninputs && data == spec.data;
  }
  bool operator!=(const ArgumentSpec & spec) const {
    return !(*this == spec);
  }
  friend struct ArgumentInfo;
  ArgumentInfo at(size_t i) const;
  size_t size() const {
    return ninputs;
  }
  size_t hashCode() const {
    return hash_code;
  }

private:
  ArrayRef<ArgumentInfoPOD> tensor_info() const {
    return ArrayRef<ArgumentInfoPOD>(reinterpret_cast<const ArgumentInfoPOD*>(data.data()), ninputs);
  }
  // the start of the sizes_strides information, which comes after the ArgumentInfoPOD list.
  const int64_t* sizes_strides() const {
    return data.data() + ninputs;
  }
  int64_t* sizes_strides() {
    return data.data() + ninputs;
  }
  size_t hash_code; // precomputed on construction
  int32_t ninputs;
  // layout is ninputs of TensorPOD (each 64-bit) followed by their size and stride info
  // for 3 tensors: [t0POD][t1POD][t2POD][t0 sizes][t0 strides][t1 sizes][t1 strides][t2 sizes][t2 strides]
  std::vector<int64_t> data;
};

// public view of compressed ArgumentInfo
struct ArgumentInfo {
  ArgumentInfo(const ArgumentSpec & spec, const int i)
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
    return (sizes_strides_offset(i + 1) - sizes_strides_offset(i))/2;
  }
  at::IntList sizes() const {
    return at::IntList(spec.sizes_strides() + sizes_strides_offset(i), ndimension());
  }
  at::IntList strides() const {
    int ndim = ndimension();
    return at::IntList(spec.sizes_strides() + sizes_strides_offset(i) + ndim, ndim);
  }
  operator TypePtr() const {
    if(!defined())
      return DynamicType::get();
    return TensorType::create(type(), device(), sizes(), strides());
  }
private:
  // offsetinto sizes_strides() array where the sizes start for tensor j
  // [valid range] valid range is [0, ninputs]
  // (i.e. you can ask for the offset at ninputs, which would be the offset of the next tensor if it existed)
  int sizes_strides_offset(int j) const {
    if(j == 0) return 0;
    return 2*pod(j - 1).total_dims;
  }
  const ArgumentInfoPOD & pod(int j) const {
    return spec.tensor_info().at(j);
  }
  const ArgumentSpec & spec;
  const int i;
};

inline std::ostream & operator<<(std::ostream & out, const ArgumentInfo & info) {
  if(!info.defined()) {
    return out << "<undefined>";
  }
  out << "Tensor(device=" << info.device()
    << ", type=" << toString(info.type())
    << ", requires_grad=" << info.requires_grad()
    << ", sizes=" << info.sizes()
    << ", strides=" << info.strides() << ")";
  return out;
}

inline std::ostream& operator<<(std::ostream & out, const ArgumentSpec & spec) {
  out << "{";
  for(size_t i = 0; i < spec.size(); ++i) {
    if (i > 0)
      out << ", ";
    out << spec.at(i);
  }
  out << "}";
  return out;
}

inline ArgumentInfo ArgumentSpec::at(size_t i) const {
  return ArgumentInfo(*this, i);
}

}}

namespace std {
  template<>
  struct hash<torch::jit::ArgumentSpec> {
    size_t operator()(const torch::jit::ArgumentSpec & spec) const {
      return spec.hashCode();
    }
  };
}

#include <iostream>
#include <vector>
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/utils/hash.h"
#include "torch/csrc/jit/variable_tensor_list.h"

namespace torch { namespace jit {

// GraphExecutor creates specializations of Graphs for different dimensionalitities
// and types of inputs.

// ArgumentSpec represents one particular specialization.
// It is designed so that it can be created, hashed, and compared quickly
// since it is used along the hot-path of the JIT to check if the code
// we have created is valid for the given inputs.

// TensorInfoPOD is only used internally in ArgumentSpec
// API users should use TensorInfo
struct TensorInfoPOD {
  // total size is 64-bit
  unsigned type : 8;
  unsigned defined : 1;
  unsigned requires_grad : 1;
  signed device : 22;
  uint32_t total_dims; // all TensorInfoPODs are in ArgumentSpec's tensor_info() array.
                       // total_dims is the total number of dimensions seen so far
                       // in all previous members of tensor_info(), including this tensor
                       // 2*total_dims becomes the offset into the sizes_strides list
                       // for the _next_ tensor in the tensor_info array
                       // for tensor 0, the offset is always 0
};

static_assert(sizeof(TensorInfoPOD) == sizeof(int64_t),
  "TensorInfoPOD must be 64-bit struct for ArgumentSpec encoding to work");

struct TensorInfo;

struct ArgumentSpec {
  // note: tensors must always be variables
  ArgumentSpec(bool with_grad, const variable_tensor_list & tensors)
  :  hash_code(0), ntensors(tensors.size()) {
    int all_dims = 0;
    for(size_t i = 0; i < ntensors; i++) {
      all_dims += tensors[i].defined() ? tensors[i].ndimension() : 0;
    }
    // allocate enough room for all TensorPODs and dimensions
    data.resize(ntensors + all_dims*2);

    // and reinterpret our data array as these structs
    TensorInfoPOD * pods = reinterpret_cast<TensorInfoPOD*>(data.data());
    int64_t * next_dim = sizes_strides();
    int total_dims = 0;
    for(size_t i = 0; i < ntensors; i++) {
      const auto & t = tensors[i];
      auto & pod = pods[i];
      pod.defined = t.defined();
      if(t.defined()) {
        pod.type = static_cast<unsigned int>(t.type().scalarType());
        pod.device = (!t.type().is_cuda()) ? -1 : t.get_device();
        pod.requires_grad = with_grad && static_cast<const autograd::Variable&>(t).requires_grad();
        total_dims += t.ndimension();
        auto sizes = t.sizes();
        std::copy(sizes.begin(),sizes.end(), next_dim);
        next_dim += sizes.size();
        auto strides = t.strides();
        std::copy(strides.begin(), strides.end(), next_dim);
        next_dim += strides.size();
      }
      // each POD has a running tally of all dimensions including its own
      pod.total_dims = total_dims;
    }
    // we precompute the hash_code to minimize the time inside of hash
    // table operations where we may need to hold a compiler cache lock.
    hash_code = hash_combine(0, ntensors);
    for(auto d : data) {
      hash_code = hash_combine(hash_code, d);
    }
  }

  // equality is fast: check ntensors, and then check the raw array data,
  // there are no size/stride indirections
  bool operator==(const ArgumentSpec & spec) const {
    return ntensors == spec.ntensors && data == spec.data;
  }
  bool operator!=(const ArgumentSpec & spec) const {
    return !(*this == spec);
  }
  friend struct TensorInfo;
  TensorInfo tensorInfo(size_t i) const;
  size_t size() const {
    return ntensors;
  }
  size_t hashCode() const {
    return hash_code;
  }

private:
  ArrayRef<TensorInfoPOD> tensor_info() const {
    return ArrayRef<TensorInfoPOD>(reinterpret_cast<const TensorInfoPOD*>(data.data()), ntensors);
  }
  // the start of the sizes_strides information, which comes after the TensorInfoPOD list.
  const int64_t* sizes_strides() const {
    return data.data() + ntensors;
  }
  int64_t* sizes_strides() {
    return data.data() + ntensors;
  }
  size_t hash_code; // precomputed on construction
  uint32_t ntensors;
  // layout is ntensors of TensorPOD (each 64-bit) followed by their size and stride info
  // for 3 tensors: [t0POD][t1POD][t2POD][t0 sizes][t0 strides][t1 sizes][t1 strides][t2 sizes][t2 strides]
  std::vector<int64_t> data;
};

// public view of compressed TensorInfo
struct TensorInfo {
  TensorInfo(const ArgumentSpec & spec, const int i)
  : spec(spec), i(i) {}
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
    return std::make_shared<TensorType>(type(), device(), sizes(), strides());
  }
private:
  // offsetinto sizes_strides() array where the sizes start for tensor j
  // [valid range] valid range is [0, ntensors]
  // (i.e. you can ask for the offset at ntensors, which would be the offset of the next tensor if it existed)
  int sizes_strides_offset(int j) const {
    if(j == 0) return 0;
    return 2*pod(j - 1).total_dims;
  }
  const TensorInfoPOD & pod(int j) const {
    return spec.tensor_info().at(j);
  }
  const ArgumentSpec & spec;
  const int i;
};

inline std::ostream & operator<<(std::ostream & out, const TensorInfo & info) {
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
    out << spec.tensorInfo(i);
  }
  return out;
}

inline TensorInfo ArgumentSpec::tensorInfo(size_t i) const {
  return TensorInfo(*this, i);
}

}}

namespace std {
  template<>
  struct hash<torch::jit::ArgumentSpec> {
    std::size_t operator()(const torch::jit::ArgumentSpec & spec) const {
      return spec.hashCode();
    }
  };
}

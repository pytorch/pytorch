#pragma once
#include <unordered_map>
#include <string>
#include <ATen/Functions.h>
#include <c10/macros/Macros.h>
#include <c10/util/irange.h>
#include <caffe2/core/context.h>
#include <caffe2/core/operator.h>
#include <caffe2/utils/math.h>
#include <iostream>

// a map from descriptor strings (see [DESCRIPTORS])
// to the key in the switch statement that implements them
static std::unordered_map<std::string, int> op_to_key = {
  ${mappings}
};

namespace caffe2 {

using at::Half; // for AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, ...)

namespace internal {
TORCH_API at::Tensor index_with_uint8_handling(
    const at::Tensor& self,
    const torch::List<c10::optional<at::Tensor>>& indices);
}

template <class Context>
class ATenOp : public Operator<Context> {
 public:
  ATenOp(const OperatorDef& operator_def, Workspace* ws)
  : Operator<Context>(operator_def, ws) {
    VLOG(2) << "ATen OpDef: " << ProtoDebugString(operator_def) << "\n";
    switch(findImplementation(operator_def)) {
      ${cases}
      default:
        CAFFE_THROW("Unexpected key value for aten operator");
    }
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    return run_op();
  }
private:
  // actual operator implementation is initialized in ctor.
  std::function<bool()> run_op;
  at::Backend backend() const;

  TypeMeta typeMetaFor(const at::Tensor & t) {
    return typeMetaFor(t.scalar_type());
  }
  TypeMeta typeMetaFor(at::ScalarType st) {
    #define DEFINE_CASE(ctype,aten_name) \
      case at::k##aten_name: \
        return TypeMeta::Make<ctype>();
    switch(st) {
      AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, DEFINE_CASE)
    default:
      CAFFE_THROW("Unknown ATen Type");
    }
    #undef DEFINE_CASE
  }

  at::TensorOptions optionsFor(const Tensor& ten) {
    at::Device device = ten.GetDevice();
#if defined(USE_ROCM)
    if (backend() == at::Backend::HIP) {
      device = at::Device(kCUDA, device.index());
    }
#endif
    return at::TensorOptions(device).dtype(ten.dtype());
  }

  at::Tensor tensorWrapping(const Tensor& ten_) {
    auto& ten = const_cast<Tensor&>(ten_);
    return at::from_blob(
        ten.raw_mutable_data(),
        ten.sizes(),
        optionsFor(ten));
  }

  at::Tensor peek(size_t i, size_t N) {
    auto real_idx = InputSize() - N + i;
    return tensorWrapping(Input(real_idx));
  }

  std::vector<at::Tensor> peekSlice(size_t i, size_t len, size_t N) {
    std::vector<at::Tensor> results;
    results.reserve(len);
    for (size_t ii = i; ii < i + len; ++ii) {
      results.push_back(peek(ii, N));
    }
    return results;
  }

  torch::List<c10::optional<at::Tensor>> peekSliceOptionals(size_t i, size_t len, size_t N) {
    torch::List<c10::optional<at::Tensor>> results;
    results.reserve(len);
    for (size_t ii = i; ii < i + len; ++ii) {
      results.push_back(peek(ii, N));
    }
    return results;
  }

  void assignTo(Tensor* dst, const at::Tensor& src_) {
    at::Tensor src = src_.contiguous();
    auto at_sizes = src.sizes();
    caffe2::TypeMeta type_meta = typeMetaFor(src);
    at::Device device = src.device();
#if defined(USE_ROCM)
    if (device.is_cuda()) {
      device = at::Device(at::DeviceType::HIP, device.index());
    }
#endif
    at::TensorImpl* src_impl = src.unsafeReleaseTensorImpl();
    std::vector<int64_t> dims(at_sizes.begin(), at_sizes.end());
    dst->Resize(dims);
    dst->ShareExternalPointer(
        at::DataPtr(
            src_impl->data(),
            static_cast<void*>(src_impl),
            [](void* t_ptr) -> void {
              at::TensorImpl* local_impl = static_cast<at::TensorImpl*>(t_ptr);
              c10::raw::intrusive_ptr::decref(local_impl);
            },
            device),
        type_meta,
        0);
  }
  void assignListStartingAt(
      size_t offset,
      const std::vector<at::Tensor>& tensors) {
    for (const auto i : c10::irange(tensors.size())) {
      assignTo(Output(offset + i), tensors[i]);
    }
  }

  template<typename T,
          typename std::enable_if<std::numeric_limits<T>::is_integer, bool>::type* =
              nullptr>
  int64_t extract(const at::Scalar &s) {
    return s.toLong();
  }

  template<typename T,
          typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type* =
              nullptr>
  int64_t extract(const at::Scalar &s) {
    return s.toDouble();
  }

  void assignTo(Tensor* dst, at::ScalarType scalar_type, const at::Scalar& scalar) {
    switch(scalar_type) {
      #define DEFINE_CASE(ctype,aten_name) \
        case at::k##aten_name: { \
          auto value = extract<ctype>(scalar); \
          assignToValue<ctype>(dst, at::convert<ctype,decltype(value)>(value)); \
        } break;
      AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, DEFINE_CASE)
#undef DEFINE_CASE
      default:
        CAFFE_THROW("Unknown ATen Type");
    }
  }
  template <typename T>
  void assignToValue(Tensor* dst, T v) {
    dst->Resize(std::vector<int64_t>());
    math::Set(1, v, dst->template mutable_data<T>(), &context_);
  }
  int findImplementation(const OperatorDef& operator_def) {
    CAFFE_ENFORCE(HasArgument("operator"));
    std::string op = OperatorBase::GetSingleArgument<std::string>("operator", "");
    // construct descriptor string ([DESCRIPTORS]) given the attributes
    // and inputs of this operator_def, and look up the implementation key
    // for this variant
    std::stringstream descriptor;
    descriptor << op;
    std::vector<std::string> attrs;
    for (const auto i : c10::irange(operator_def.arg_size())) {
      auto & attr = operator_def.arg(i);
      if(attr.name() == "operator" || attr.name() == "type" )
        continue;
      attrs.push_back(attr.name());
    }
    std::sort(attrs.begin(), attrs.end());
    for(auto & a : attrs)
      descriptor << "-" << a;

    std::string descriptor_sized =
        descriptor.str() + "-" + c10::to_string(InputSize());
    std::string descriptor_var_args = descriptor.str() + "-*";
    if (op_to_key.count(descriptor_sized) > 0) {
      return op_to_key[descriptor_sized];
    }
    if (op_to_key.count(descriptor_var_args) > 0) {
      return op_to_key[descriptor_var_args];
    }
    std::stringstream ss;
    ss << "Attempting to run unknown ATen operator configuration: "
       << descriptor_sized;
    CAFFE_THROW(ss.str());
  }
  at::Scalar readScalarAttribute(const std::string & name) {
    if(OperatorBase::HasSingleArgumentOfType<int64_t>(name)) {
      return OperatorBase::GetSingleArgument<int64_t>(name, 0);
    } else {
      CAFFE_ENFORCE(OperatorBase::HasSingleArgumentOfType<float>(name));
      return OperatorBase::GetSingleArgument<float>(name, 0);
    }
  }
  template<typename T>
  T readAttribute(const std::string & name) {
    CAFFE_ENFORCE(OperatorBase::HasSingleArgumentOfType<T>(name));
    return OperatorBase::GetSingleArgument<T>(name, 0);
  }
  std::vector<int64_t> readIntArrayRef(const std::string & name) {
    CAFFE_ENFORCE(OperatorBase::HasArgument(name));
    return OperatorBase::GetRepeatedArgument<int64_t>(name, {});
  }
  template <int N>
  std::array<bool, N> readBoolMask(const std::string& name) {
    CAFFE_ENFORCE(OperatorBase::HasArgument(name));
    std::vector<int64_t> ints =
        OperatorBase::GetRepeatedArgument<int64_t>(name, {});
    std::array<bool, N> result;
    for (const auto i : c10::irange(N)) {
      result[i] = ints.at(i);
    }
    return result;
  }

  ${implementations}
};

}

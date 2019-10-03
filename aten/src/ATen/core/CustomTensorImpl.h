#pragma once

#include <string>
#include <unordered_map>
#include <tuple>
#include <c10/util/C++17.h>
#include <ATen/core/stack.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/TensorImpl.h>

/*
  The API is effectively

  // Register functions, either raw or one that takes a function schema and stack
  REGISTER_CUSTOM_TENSOR_METHOD(name, method, impl)
  REGISTER_CUSTOM_TENSOR_STACK_METHOD(name, method, impl)

  // Get the allocated tag from the name of the custom tensor
  GET_CUSTOM_TENSOR_TAG(name)

  // Extract a pointer to your tensor from the generic at::Tensor
  template <typename CustomTensor>
  std::shared_ptr<CustomTensor> extractCustomTensor(at::Tensor t);
*/

using Impl = void(const c10::FunctionSchema& schema, torch::jit::Stack*);

namespace at {

class CAFFE2_API CustomTensorImpl : public TensorImpl {
  size_t tag_;
  std::shared_ptr<void> storage_;
 public:
  inline size_t tag() {
    return tag_;
  }
  inline std::shared_ptr<void> storage() {
    return storage_;
  }
  CustomTensorImpl(
    size_t tag, std::shared_ptr<void> storage, TensorTypeSet ts,
    caffe2::TypeMeta dtype, c10::Device device) :
    TensorImpl(ts.add(TensorTypeId::CustomTensorId),
        dtype,
        device) {
        tag_ = tag;
        storage_ = storage;
        }
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override;
};

struct CAFFE2_API RegisterCustomTensorImpl {
  RegisterCustomTensorImpl(std::string name, std::string method, Impl* impl);
};

CAFFE2_API std::unordered_map<std::string, size_t>& getTagNameMap();

}

namespace detail {

template<typename T, std::size_t... I>
T unbox__(const std::vector<c10::IValue>& a, c10::guts::index_sequence<I...>) {
    return std::make_tuple(a[I].to< typename std::tuple_element<I, T>::type >()...);
}

template<typename T, typename Indices = c10::guts::make_index_sequence<std::tuple_size<T>::value>>
T unbox_(const std::vector<c10::IValue>& args) {
    return unbox__<T>(args, Indices{});
}

template<typename... T>
std::tuple<T...> unbox(const std::vector<c10::IValue>& args) {
    return unbox_<std::tuple<T...>>(args);
}

template <typename R, typename... Args>
void call(R(f)(Args...), torch::jit::Stack* s) {
    static_assert(!c10::guts::disjunction<std::is_reference<Args>...>::value,
      "Drop references (&) from the args of your function.");
    std::vector<c10::IValue> args = torch::jit::pop(*s, sizeof...(Args));
    torch::jit::push(*s, c10::guts::apply(f, unbox<Args...>(args)));
}

} // namespace detail

#define REGISTER_CUSTOM_TENSOR_STACK_METHOD(name, method, impl) \
  static at::RegisterCustomTensorImpl implementations_##name_##method(#name, #method, impl);

#define REGISTER_CUSTOM_TENSOR_METHOD(name, method, impl) \
  static at::RegisterCustomTensorImpl implementations_##name_##method(#name, #method, \
      [](const FunctionSchema&, torch::jit::Stack* s) -> void {\
        ::detail::call(impl, s);\
      });

#define GET_CUSTOM_TENSOR_TAG(name) \
  at::getTagNameMap()[name];

namespace at {

template <typename CustomTensor>
std::shared_ptr<CustomTensor> extractCustomTensor(at::Tensor t) {
  TORCH_CHECK(t.type_set().has(TensorTypeId::CustomTensorId), "Passed in tensor isn't CustomTensor");
  auto cst = static_cast<CustomTensorImpl*>(t.unsafeGetTensorImpl());
  return std::static_pointer_cast<CustomTensor>(cst->storage());
}

} // namespace at


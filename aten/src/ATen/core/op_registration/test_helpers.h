#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <ATen/core/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <c10/core/CPUAllocator.h>

namespace detail {
// InputToIValue takes a value and converts it to an IValue to be put on a stack.
template<class T>
struct InputToIValue final {
  template<class T_>
  static c10::IValue call(T_&& v) {
    return c10::IValue(std::forward<T_>(v));
  }
};
template<class T>
struct InputToIValue<c10::optional<T>> final {
  template<class T_>
  static c10::IValue call(T_&& v) {
    if (v.has_value()) {
      return c10::IValue(std::move(*v));
    } else {
      return c10::IValue();
    }
  }
};
template<class T>
struct InputToIValue<c10::ArrayRef<T>> final {
  template<class T_>
  static c10::IValue call(T_&& v) {
    return c10::IValue(v.vec());
  }
};
template<class Key, class Value>
struct InputToIValue<std::vector<std::unordered_map<Key, Value>>> final {
  template<class T_>
  static c10::IValue call(T_&& v) {
    auto list = c10::ivalue::GenericList::create({});
    list->elements().reserve(v.size());
    for (std::unordered_map<Key, Value>& e : v) {
      list->elements().push_back(InputToIValue<std::unordered_map<Key, Value>>::call(std::move(e)));
    }
    return list;
  }
};
template<>
struct InputToIValue<std::vector<std::string>> final {
  template<class T_>
  static c10::IValue call(T_&& v) {
    auto list = c10::ivalue::GenericList::create({});
    list->elements().reserve(v.size());
    for (std::string& e : v) {
      list->elements().push_back(InputToIValue<std::string>::call(std::move(e)));
    }
    return list;
  }
};
template<class Key, class Value>
struct InputToIValue<c10::Dict<Key, Value>> final {
  template<class T_>
  static c10::IValue call(T_&& v) {
    return c10::IValue(c10::impl::toGenericDict(std::move(v)));
  }
};
template<class Key, class Value>
struct InputToIValue<std::unordered_map<Key, Value>> final {
  template<class T_>
  static c10::IValue call(T_&& v) {
    c10::impl::GenericDict dict;
    dict.reserve(v.size());
    for (auto& element : v) {
      dict.insert(InputToIValue<Key>::call(element.first), InputToIValue<Value>::call(element.second));
    }
    return c10::IValue(std::move(dict));
  }
};
}

template<class... Inputs>
inline std::vector<c10::IValue> makeStack(Inputs&&... inputs) {
  return {detail::InputToIValue<c10::guts::decay_t<Inputs>>::call(std::forward<Inputs>(inputs))...};
}

inline at::Tensor dummyTensor(c10::TensorTypeId dispatch_key) {
  auto* allocator = c10::GetCPUAllocator();
  int64_t nelements = 1;
  auto dtype = caffe2::TypeMeta::Make<float>();
  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
    dtype,
    nelements,
    allocator->allocate(nelements * dtype.itemsize()),
    allocator,
    /*resizable=*/true);
  return at::detail::make_tensor<c10::TensorImpl>(storage_impl, dispatch_key);
}

template<class... Args>
inline std::vector<c10::IValue> callOp(const c10::OperatorHandle& op, Args... args) {
  auto stack = makeStack(std::forward<Args>(args)...);
  auto kernel = c10::Dispatcher::singleton().lookup(op, &stack);
  kernel.call(&stack);
  return stack;
}

inline void expectDoesntFindKernel(const char* op_name, c10::TensorTypeId dispatch_key) {
  auto op = c10::Dispatcher::singleton().findSchema(op_name, "");
  EXPECT_ANY_THROW(
    callOp(*op, dummyTensor(dispatch_key), 5);
  );
}

inline void expectDoesntFindOperator(const char* op_name) {
  auto op = c10::Dispatcher::singleton().findSchema(op_name, "");
  EXPECT_FALSE(op.has_value());
}

template<class Exception, class Functor>
inline void expectThrows(Functor&& functor, const char* expectMessageContains) {
  try {
    std::forward<Functor>(functor)();
  } catch (const Exception& e) {
    EXPECT_THAT(e.what(), testing::HasSubstr(expectMessageContains));
    return;
  }
  ADD_FAILURE() << "Expected to throw exception containing \""
    << expectMessageContains << "\" but didn't throw";
}

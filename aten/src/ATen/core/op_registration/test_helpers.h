#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <ATen/core/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <c10/core/CPUAllocator.h>

template<class... Inputs>
inline std::vector<c10::IValue> makeStack(Inputs&&... inputs) {
  return {std::forward<Inputs>(inputs)...};
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
  auto op = c10::Dispatcher::singleton().findSchema({op_name, ""});
  EXPECT_ANY_THROW(
    callOp(*op, dummyTensor(dispatch_key), 5);
  );
}

inline void expectDoesntFindOperator(const char* op_name) {
  auto op = c10::Dispatcher::singleton().findSchema({op_name, ""});
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

template<class T>
void expectListEquals(c10::ArrayRef<T> expected, c10::List<T> actual) {
  EXPECT_EQ(expected.size(), actual.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(expected[i], actual.get(i));
  }
}

template<class T>
void expectListEquals(c10::ArrayRef<T> expected, std::vector<T> actual) {
  EXPECT_EQ(expected.size(), actual.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(expected[i], actual[i]);
  }
}

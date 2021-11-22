#ifndef COMPUTATION_CLIENT_UTIL_H_
#define COMPUTATION_CLIENT_UTIL_H_

#include <c10/util/Optional.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

#include "torch/csrc/lazy/core/hash.h"

namespace torch {
namespace lazy {
template<typename T>
c10::optional<std::vector<T>> ToOptionalVector(c10::optional<c10::ArrayRef<T>> arrayRef) {
  if (arrayRef) {
    return arrayRef->vec();
  }
  return c10::nullopt;
}
}  // namespace lazy
}  // namespace torch

namespace lazy_tensors {
namespace util {

// Similar to c10::scope_exit but with a status.
// TODO(alanwaketan): Consolidate it with c10::scope_exit.
template <typename T>
class Cleanup {
 public:
  using StatusType = T;

  explicit Cleanup(std::function<void(StatusType)> func)
      : func_(std::move(func)) {}
  Cleanup(Cleanup&& ref)
      : func_(std::move(ref.func_)), status_(std::move(ref.status_)) {}
  Cleanup(const Cleanup&) = delete;

  ~Cleanup() {
    if (func_ != nullptr) {
      func_(std::move(status_));
    }
  }

  Cleanup& operator=(const Cleanup&) = delete;

  Cleanup& operator=(Cleanup&& ref) {
    if (this != &ref) {
      func_ = std::move(ref.func_);
      status_ = std::move(ref.status_);
    }
    return *this;
  }

  void Release() { func_ = nullptr; }

  void SetStatus(StatusType status) { status_ = std::move(status); }

  const StatusType& GetStatus() const { return status_; }

 private:
  std::function<void(StatusType)> func_;
  StatusType status_;
};

using ExceptionCleanup = Cleanup<std::exception_ptr>;

// Allows APIs which might return const references and values, to not be forced
// to return values in the signature.
// TODO(alanwaketan): This is clever, but is there really no std or c10 supports?
// Needs more investigations.
template <typename T>
class MaybeRef {
 public:
  MaybeRef(const T& ref) : ref_(ref) {}
  MaybeRef(T&& value) : storage_(std::move(value)), ref_(*storage_) {}

  const T& get() const { return ref_; }

  const T& operator*() const { return get(); }

  operator const T&() const { return get(); }

  bool is_stored() const { return storage_.has_value(); }

 private:
  c10::optional<T> storage_;
  const T& ref_;
};

template <typename T>
std::vector<T> Iota(size_t size, T init = 0, T incr = 1) {
  std::vector<T> result(size);
  T value = init;
  for (size_t i = 0; i < size; ++i, value += incr) {
    result[i] = value;
  }
  return result;
}

template <typename T, typename S>
std::vector<T> ToVector(const S& input) {
  return std::vector<T>(input.begin(), input.end());
}

template <typename T>
typename std::underlying_type<T>::type GetEnumValue(T value) {
  return static_cast<typename std::underlying_type<T>::type>(value);
}

// The following is only used within computation_client.
template <typename T, typename G>
const typename T::mapped_type& MapInsert(T* cont,
                                         const typename T::key_type& key,
                                         const G& gen) {
  auto it = cont->find(key);
  if (it == cont->end()) {
    it = cont->emplace(key, gen()).first;
  }
  return it->second;
}

}  // namespace util
}  // namespace lazy_tensors

#endif  // COMPUTATION_CLIENT_UTIL_H_

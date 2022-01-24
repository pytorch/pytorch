#ifndef CAFFE2_OPERATORS_INDEX_OPS_H_
#define CAFFE2_OPERATORS_INDEX_OPS_H_

#include <limits>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>
#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "c10/util/irange.h"

namespace caffe2 {
namespace {
using IndexKeyTypes = TensorTypes<int32_t, int64_t, std::string>;
using int64_tValue = int64_t;
} // namespace

struct IndexBase {
 public:
  IndexBase(int64_tValue maxElements, const TypeMeta type)
      : maxElements_{maxElements}, meta_(type), frozen_{false} {}

  void Freeze() {
    frozen_ = true;
  }

  bool isFrozen() const {
    return frozen_;
  }

  int64_t maxElements() const {
    return maxElements_;
  }

  virtual ~IndexBase() {}

  const TypeMeta Type() const {
    return meta_;
  }

  int64_tValue Size() {
    std::lock_guard<std::mutex> guard(dictMutex_);
    return nextId_;
  }

 protected:
  int64_t maxElements_;
  TypeMeta meta_;
  int64_tValue nextId_{1}; // guarded by dictMutex_
  std::atomic<bool> frozen_{false};
  std::mutex dictMutex_;
};

template <typename T>
struct Index : IndexBase {
  explicit Index(int64_tValue maxElements)
      : IndexBase(maxElements, TypeMeta::Make<T>()) {}

  void Get(const T* keys, int64_tValue* values, size_t numKeys) {
    if (frozen_) {
      FrozenGet(keys, values, numKeys);
      return;
    }
    std::lock_guard<std::mutex> lock(dictMutex_);
    // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
    for (const auto i : c10::irange(numKeys)) {
      auto it = dict_.find(keys[i]);
      if (it != dict_.end()) {
        values[i] = it->second;
      } else if (nextId_ < maxElements_) {
        auto newValue = nextId_++;
        dict_.insert({keys[i], newValue});
        values[i] = newValue;
      } else {
        CAFFE_THROW("Dict max size reached");
      }
    }
  }

  bool Load(const T* keys, size_t numKeys) {
    CAFFE_ENFORCE(
        // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
        numKeys <= maxElements_,
        "Cannot load index: Tensor is larger than max_elements.");
    decltype(dict_) dict;
    for (const auto i : c10::irange(0U, numKeys)) {
      CAFFE_ENFORCE(
          dict.insert({keys[i], i + 1}).second,
          "Repeated elements found: cannot load into dictionary.");
    }
    // assume no `get` is inflight while this happens
    {
      std::lock_guard<std::mutex> lock(dictMutex_);
      // let the old dict get destructed outside of the lock
      dict_.swap(dict);
      nextId_ = numKeys + 1;
    }
    return true;
  }

  bool Store(Tensor* out) {
    std::lock_guard<std::mutex> lock(dictMutex_);
    out->Resize(nextId_ - 1);
    auto outData = out->template mutable_data<T>();
    for (const auto& entry : dict_) {
      outData[entry.second - 1] = entry.first;
    }
    return true;
  }

 private:
  void FrozenGet(const T* keys, int64_tValue* values, size_t numKeys) {
    for (const auto i : c10::irange(0U, numKeys)) {
      auto it = dict_.find(keys[i]);
      values[i] = it != dict_.end() ? it->second : 0;
    }
  }

  std::unordered_map<T, int64_tValue> dict_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INDEX_OPS_H_

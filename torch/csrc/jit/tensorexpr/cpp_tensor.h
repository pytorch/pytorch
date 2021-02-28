#pragma once

namespace torch {
namespace jit {
namespace tensorexpr {

// TODO: use SIMD such as AVX
constexpr auto cpp_tensor_definition = R"(

template <typename T>
class Tensor {
 public:
  explicit Tensor(const std::array<int>& dims) : strides_(dims.size()) {
    int size = sizeof(T);
    for (int dim : dims) {
      size *= dim;
    }
    storage_ = static_cast<T*>(malloc(size));

    int ndim = dims.size();
    strides_[ndim - 1] = 1;
    for (size_t i = ndim - 2; i >= 0; i--) {
      strides_[i] = strides[i + 1] * dims[i + 1];
    }
  }

  ~Tensor() {
    free();
  }

  void free() {
    if (storage_ != nullptr) {
      free(storage_);
      storage_ = nullptr;
    }
  }

  const T& operator[](const std::array<int>& idx) const {
    return storage_[flatten_index(idx)];
  }

  T& operator[](const std::array<int>& idx) {
    return storage_[flatten_index(idx)];
  }

  Vector<T> load(const Vector<int>& idx, const Vector<int>& mask) const {
    assert(strides_.size() == 1);
    assert(idx.len() == mask.len());
    DenseVector<T> res(idx.len());
    for (size_t i = 0; i < idx.len(); i++) {
      if (mask[i]) {
        res[i] = storage_[idx[i]];
      }
    }
    return res;
  }

  void store(const Vector<int>& idx, const Vector<T>& value, const Vector<int>& mask) {
    assert(strides_.size() == 1);
    assert(idx.len() == value.len());
    assert(idx.len() == mask.len());
    for (size_t i = 0; i < idx.len(); i++) {
      if (mask[i]) {
        storage_[idx[i]] = value[i];
      }
    }
  }

 private:
  T* storage_;
  std::array<int> strides_;

  int flatten_index(const std::array<int>& idx) const {
    int flat_idx = 0;
    for (size_t i = 0; i < strides_.size(); i++) {
      flat_idx += idx[i] * strides_[i];
    }
    return flat_idx;
  }
};

)";

} // namespace tensorexpr
} // namespace jit
} // namespace torch

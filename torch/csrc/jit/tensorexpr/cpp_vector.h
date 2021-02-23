#pragma once

namespace torch {
namespace jit {
namespace tensorexpr {

// TODO: use SIMD such as AVX
constexpr auto cpp_vector_definition = R"(

template <typename T>
class Vector {
 public:
  virtual ~Vector() {}

  virtual size_t len() const = 0;

  virtual const T& operator[](size_t idx) const = 0;
};

template <typename T>
class DenseVector : public Vector<T> {
 public:
  DenseVector(size_t len) : vec_(len, 0) {}

  size_t len() const override {
    return vec_.size();
  }

  const T& operator[](size_t idx) const override {
    assert(idx >= 0 && idx < vec_.size());
    return vec_.at(idx);
  }

  T& operator[](size_t idx) {
    assert(idx >= 0 && idx < vec_.size());
    return vec_[idx];
  }

 private:
  std::vector<T> vec_;
};

class Ramp : public Vector<int> {
 public:
  Ramp(int base, int stride, int lanes)
      : base_(base), stride_(stride), lanes_(lanes) {}

  size_t len() const override {
    return static_cast<size_t>(lanes_);
  }

  const int& operator[](size_t idx) const override {
    assert(idx >= 0 && idx < static_cast<size_t>(lanes_));
    return base_ + stride_ * static_cast<int>(idx);
  }

 private:
  int base_;
  int stride_;
  int lanes_;
};

template <typename T>
class Broadcast : public Vector<T> {
 public:
  Broadcast(T v, int lanes) : v_(v), lanes_(lanes) {}

  size_t len() const override {
    return static_cast<size_t>(lanes_);
  }

  const T& operator[](size_t idx) const override {
    assert(idx >= 0 && idx < static_cast<size_t>(lanes_));
    return v_;
  }

 private:
  T v_;
  int lanes_;
};

template <typename TInput, typename TReturn>
Vector<TReturn> vectorizedUnaryOp(const Vector<TInput>& v,
    std::function<TReturn(TInput)> unary_op) const {
  DenseVector<TReturn> res(v.len());
  for (size_t i = 0; i < v.len(); i++) {
    res[i] = unary_op(v[i]);
  }
  return res;
}

template <typename T>
Vector<T> vectorizedBinaryOp(const Vector<T>& a, const Vector<T>& b,
    std::function<T(T, T)> binary_op) const {
  assert(a.len() == b.len());
  DenseVector<T> res(a.len());
  for (size_t i = 0; i < a.len(); i++) {
    res[i] = binary_op(a[i], b[i]);
  }
  return res;
}

template <typename T>
Vector<T> operator+(const Vector<T>& lhs, const Vector<T>& rhs) const {
  return vectorizedBinaryOp(lhs, rhs, [](T a, T b) { return a + b; });
}

template <typename T>
Vector<T> operator-(const Vector<T>& lhs, const Vector<T>& rhs) const {
  return vectorizedBinaryOp(lhs, rhs, [](T a, T b) { return a - b; });
}

template <typename T>
Vector<T> operator*(const Vector<T>& lhs, const Vector<T>& rhs) const {
  return vectorizedBinaryOp(lhs, rhs, [](T a, T b) { return a * b; });
}

template <typename T>
Vector<T> operator/(const Vector<T>& lhs, const Vector<T>& rhs) const {
  return vectorizedBinaryOp(lhs, rhs, [](T a, T b) { return a / b; });
}

template <typename T>
Vector<T> operator%(const Vector<T>& lhs, const Vector<T>& rhs) const {
  return vectorizedBinaryOp(lhs, rhs, [](T a, T b) {
    if (std::is_floating_point<T>::value) {
      return std::fmod(a, b);
    } else if (std::is_integral<T>::value) {
      return a % b;
    } else {
      throw std::runtime_error("Modula arithmetic can only happen for integer and floating point values");
    }
  });
}

template <typename T>
Vector<T> operator&(const Vector<T>& lhs, const Vector<T>& rhs) const {
  return vectorizedBinaryOp(lhs, rhs, [](T a, T b) { return a & b; });
}

template <typename T>
Vector<T> operator|(const Vector<T>& lhs, const Vector<T>& rhs) const {
  return vectorizedBinaryOp(lhs, rhs, [](T a, T b) { return a | b; });
}

template <typename T>
Vector<T> operator^(const Vector<T>& lhs, const Vector<T>& rhs) const {
  return vectorizedBinaryOp(lhs, rhs, [](T a, T b) { return a ^ b; });
}

template <typename T>
Vector<T> operator<<(const Vector<T>& lhs, const Vector<T>& rhs) const {
  return vectorizedBinaryOp(lhs, rhs, [](T a, T b) { return a << b; });
}

template <typename T>
Vector<T> operator>>(const Vector<T>& lhs, const Vector<T>& rhs) const {
  return vectorizedBinaryOp(lhs, rhs, [](T a, T b) { return a >> b; });
}

template <typename T>
typename std::enable_if_t<std::is_floating_point<T>::value || std::is_integral<T>::value, T> max_value(
    T a,
    T b) {
  return std::max(a, b);
}

template <typename T>
typename std::enable_if_t<!std::is_floating_point<T>::value && !std::is_integral<T>::value, T> max_value(
    T a,
    T b) {
  return a < b ? b : a;
}

template <typename T>
Vector<T> Max(const Vector<T>& lhs, const Vector<T>& rhs) {
  return vectorizedBinaryOp(lhs, rhs, [](T a, T b) {
    return max_value(a, b);
  });
}

template <typename T>
typename std::enable_if_t<std::is_floating_point<T>::value || std::is_integral<T>::value, T> min_value(
    T a,
    T b) {
  return std::min(a, b);
}

template <typename T>
typename std::enable_if_t<!std::is_floating_point<T>::value && !std::is_integral<T>::value, T> min_value(
    T a,
    T b) {
  return a < b ? a : b;
}

template <typename T>
Vector<T> Min(const Vector<T>& lhs, const Vector<T>& rhs) {
  return vectorizedBinaryOp(lhs, rhs, [](T a, T b) {
    return min_value(a, b);
  });
}

template <typename From, typename To>
Vector<To> Cast(const Vector<From>& v) {
  return vectorizedUnaryOp<From, To>(v, [](From v) {
    return static_cast<To>(v);
  });
}

template <typename From, typename To>
To BitCast(const From& v) {
  assert(sizeof(To) == sizeof(From));
  To res;
  std::memcpy(&res, &v, sizeof(From));
  return res;
}

template <typename From, typename To>
Vector<To> BitCast(const Vector<From>& v) {
  return vectorizedUnaryOp<From, To>(v, [](From v) {
    return BitCast<From, To>(v);
  });
}

template <typename TInput, typename TReturn>
Vector<TReturn> CompareSelect(std::function<bool(TInput, TInput)> cmp,
    const Vector<TInput>& lhs, const Vector<TInput>& rhs,
    const Vector<TReturn>& true_result, const Vector<TReturn>& false_result) {
  assert(lhs.len() == rhs.len());
  DenseVector<TReturn> res(lhs.len());
  for (size_t i = 0; i < lhs.len(); i++) {
    res[i] = cmp(lhs[i], rhs[i]) ? true_result[i] : false_result[i];
  }
  return res;
}

)";

} // namespace tensorexpr
} // namespace jit
} // namespace torch

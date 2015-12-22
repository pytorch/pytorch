// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_META_H
#define EIGEN_CXX11_TENSOR_TENSOR_META_H

namespace Eigen {

template<bool cond> struct Cond {};

template<typename T1, typename T2> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
const T1& choose(Cond<true>, const T1& first, const T2&) {
  return first;
}

template<typename T1, typename T2> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
const T2& choose(Cond<false>, const T1&, const T2& second) {
  return second;
}

template <size_t n> struct max_n_1 {
  static const size_t size = n;
};
template <> struct max_n_1<0> {
  static const size_t size = 1;
};


// Default packet types
template <typename Scalar, typename Device>
struct PacketType {
  typedef typename internal::packet_traits<Scalar>::type type;
  static const int size = internal::unpacket_traits<type>::size;
};

// For CUDA packet types when using a GpuDevice
#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
template <>
struct PacketType<float, GpuDevice> {
  typedef float4 type;
  static const int size = 4;
};
template <>
struct PacketType<double, GpuDevice> {
  typedef double2 type;
  static const int size = 2;
};
#endif



// Tuple mimics std::pair but works on e.g. nvcc.
template <typename U, typename V> struct Tuple {
 public:
  U first;
  V second;

  typedef U first_type;
  typedef V second_type;

  EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  Tuple() : first(), second() {}

  EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  Tuple(const U& f, const V& s) : first(f), second(s) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  Tuple& operator= (const Tuple& rhs) {
    if (&rhs == this) return *this;
    first = rhs.first;
    second = rhs.second;
    return *this;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  void swap(Tuple& rhs) {
    using numext::swap;
    swap(first, rhs.first);
    swap(second, rhs.second);
  }
};

template <typename U, typename V>
EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
bool operator==(const Tuple<U, V>& x, const Tuple<U, V>& y) {
  return (x.first == y.first && x.second == y.second);
}

template <typename U, typename V>
EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
bool operator!=(const Tuple<U, V>& x, const Tuple<U, V>& y) {
  return !(x == y);
}



#ifdef EIGEN_HAS_SFINAE
namespace internal{

  template<typename IndexType, Index... Is>
  EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  array<Index, sizeof...(Is)> customIndices2Array(IndexType& idx, numeric_list<Index, Is...>) {
    return { idx[Is]... };
  }

  /** Make an array (for index/dimensions) out of a custom index */
  template<typename Index, std::size_t NumIndices, typename IndexType>
  EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  array<Index, NumIndices> customIndices2Array(IndexType& idx) {
    return customIndices2Array(idx, typename gen_numeric_list<Index, NumIndices>::type{});
  }


  template <typename B, typename D>
  struct is_base_of
  {

    typedef char (&yes)[1];
    typedef char (&no)[2];

    template <typename BB, typename DD>
    struct Host
    {
      operator BB*() const;
      operator DD*();
    };

    template<typename T>
    static yes check(D*, T);
    static no check(B*, int);

    static const bool value = sizeof(check(Host<B,D>(), int())) == sizeof(yes);
  };

}
#endif



}  // namespace Eigen

#endif  // EIGEN_CXX11_TENSOR_TENSOR_META_H

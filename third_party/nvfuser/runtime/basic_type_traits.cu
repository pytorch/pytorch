#ifdef __NVCC__
#include <type_traits>
#else
// The following namespace std is modified from LLVM, see the following copyright information
//
// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// copy-pasted from some llvm files:
// - https://github.com/llvm/llvm-project/blob/main/libcxx/include/type_traits
// - https://github.com/llvm/llvm-project/blob/main/clang/test/Headers/Inputs/include/type_traits
namespace std {

  template <class _Tp>
  _Tp&& __declval(int);
  template <class _Tp>
  _Tp __declval(long);
  template <class _Tp>
  decltype(__declval<_Tp>(0)) declval() noexcept;
  
  template <class _Tp, _Tp __v>
  struct integral_constant {
    static const _Tp value = __v;
    typedef _Tp value_type;
    typedef integral_constant type;
  };
  
  typedef integral_constant<bool, true> true_type;
  typedef integral_constant<bool, false> false_type;
  
  // is_same, functional
  template <class _Tp, class _Up>
  struct is_same : public false_type {};
  template <class _Tp>
  struct is_same<_Tp, _Tp> : public true_type {};
  
  // is_integral, for some types.
  template <class _Tp>
  struct is_integral : public integral_constant<bool, false> {};
  template <>
  struct is_integral<bool> : public integral_constant<bool, true> {};
  template <>
  struct is_integral<char> : public integral_constant<bool, true> {};
  template <>
  struct is_integral<short> : public integral_constant<bool, true> {};
  template <>
  struct is_integral<int> : public integral_constant<bool, true> {};
  template <>
  struct is_integral<long> : public integral_constant<bool, true> {};
  template <>
  struct is_integral<long long> : public integral_constant<bool, true> {};
  
  // enable_if, functional
  template <bool _C, typename _Tp>
  struct enable_if {};
  template <typename _Tp>
  struct enable_if<true, _Tp> {
    using type = _Tp;
  };
  template <bool b, class T = void>
  using enable_if_t = typename enable_if<b, T>::type;
  
  template <class _Tp>
  struct remove_const {
    typedef _Tp type;
  };
  template <class _Tp>
  struct remove_const<const _Tp> {
    typedef _Tp type;
  };
  template <class _Tp>
  using remove_const_t = typename remove_const<_Tp>::type;
  
  template <class _Tp>
  struct remove_volatile {
    typedef _Tp type;
  };
  template <class _Tp>
  struct remove_volatile<volatile _Tp> {
    typedef _Tp type;
  };
  template <class _Tp>
  using remove_volatile_t = typename remove_volatile<_Tp>::type;
  
  template <class _Tp>
  struct remove_cv {
    typedef typename remove_volatile<typename remove_const<_Tp>::type>::type type;
  };
  template <class _Tp>
  using remove_cv_t = typename remove_cv<_Tp>::type;
  
  template <class _Tp>
  struct __libcpp_is_floating_point : public false_type {};
  template <>
  struct __libcpp_is_floating_point<float> : public true_type {};
  template <>
  struct __libcpp_is_floating_point<double> : public true_type {};
  template <>
  struct __libcpp_is_floating_point<long double> : public true_type {};
  
  template <class _Tp>
  struct is_floating_point
      : public __libcpp_is_floating_point<typename remove_cv<_Tp>::type> {};
  
  template <class _Tp>
  struct is_arithmetic
      : public integral_constant<
            bool,
            is_integral<_Tp>::value || is_floating_point<_Tp>::value> {};
  template <class _Tp>
  inline constexpr bool is_arithmetic_v = is_arithmetic<_Tp>::value;
  
  template <class _Tp>
  struct __numeric_type {
    static void __test(...);
    static float __test(float);
    static double __test(char);
    static double __test(int);
    static double __test(unsigned);
    static double __test(long);
    static double __test(unsigned long);
    static double __test(long long);
    static double __test(unsigned long long);
    static double __test(double);
    static long double __test(long double);
  
    typedef decltype(__test(declval<_Tp>())) type;
    static const bool value = !is_same<type, void>::value;
  };
  
  template <>
  struct __numeric_type<void> {
    static const bool value = true;
  };
  
  // __promote
  
  template <
      class _A1,
      class _A2 = void,
      class _A3 = void,
      bool = __numeric_type<_A1>::value&& __numeric_type<_A2>::value&&
          __numeric_type<_A3>::value>
  class __promote_imp {
   public:
    static const bool value = false;
  };
  
  template <class _A1, class _A2, class _A3>
  class __promote_imp<_A1, _A2, _A3, true> {
   private:
    typedef typename __promote_imp<_A1>::type __type1;
    typedef typename __promote_imp<_A2>::type __type2;
    typedef typename __promote_imp<_A3>::type __type3;
  
   public:
    typedef decltype(__type1() + __type2() + __type3()) type;
    static const bool value = true;
  };
  
  template <class _A1, class _A2>
  class __promote_imp<_A1, _A2, void, true> {
   private:
    typedef typename __promote_imp<_A1>::type __type1;
    typedef typename __promote_imp<_A2>::type __type2;
  
   public:
    typedef decltype(__type1() + __type2()) type;
    static const bool value = true;
  };
  
  template <class _A1>
  class __promote_imp<_A1, void, void, true> {
   public:
    typedef typename __numeric_type<_A1>::type type;
    static const bool value = true;
  };
  
  template <class _A1, class _A2 = void, class _A3 = void>
  class __promote : public __promote_imp<_A1, _A2, _A3> {};
  
  } // namespace std
#endif
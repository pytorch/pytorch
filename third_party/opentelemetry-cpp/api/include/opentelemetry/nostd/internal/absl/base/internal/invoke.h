// Copyright 2017 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// absl::base_internal::Invoke(f, args...) is an implementation of
// INVOKE(f, args...) from section [func.require] of the C++ standard.
//
// [func.require]
// Define INVOKE (f, t1, t2, ..., tN) as follows:
// 1. (t1.*f)(t2, ..., tN) when f is a pointer to a member function of a class T
//    and t1 is an object of type T or a reference to an object of type T or a
//    reference to an object of a type derived from T;
// 2. ((*t1).*f)(t2, ..., tN) when f is a pointer to a member function of a
//    class T and t1 is not one of the types described in the previous item;
// 3. t1.*f when N == 1 and f is a pointer to member data of a class T and t1 is
//    an object of type T or a reference to an object of type T or a reference
//    to an object of a type derived from T;
// 4. (*t1).*f when N == 1 and f is a pointer to member data of a class T and t1
//    is not one of the types described in the previous item;
// 5. f(t1, t2, ..., tN) in all other cases.
//
// The implementation is SFINAE-friendly: substitution failure within Invoke()
// isn't an error.

#ifndef OTABSL_BASE_INTERNAL_INVOKE_H_
#define OTABSL_BASE_INTERNAL_INVOKE_H_

#include <algorithm>
#include <type_traits>
#include <utility>

#include "../../meta/type_traits.h"

// The following code is internal implementation detail.  See the comment at the
// top of this file for the API documentation.

namespace absl {
OTABSL_NAMESPACE_BEGIN
namespace base_internal {

// The five classes below each implement one of the clauses from the definition
// of INVOKE. The inner class template Accept<F, Args...> checks whether the
// clause is applicable; static function template Invoke(f, args...) does the
// invocation.
//
// By separating the clause selection logic from invocation we make sure that
// Invoke() does exactly what the standard says.

template <typename Derived>
struct StrippedAccept {
  template <typename... Args>
  struct Accept : Derived::template AcceptImpl<typename std::remove_cv<
                      typename std::remove_reference<Args>::type>::type...> {};
};

// (t1.*f)(t2, ..., tN) when f is a pointer to a member function of a class T
// and t1 is an object of type T or a reference to an object of type T or a
// reference to an object of a type derived from T.
struct MemFunAndRef : StrippedAccept<MemFunAndRef> {
  template <typename... Args>
  struct AcceptImpl : std::false_type {};

  template <typename MemFunType, typename C, typename Obj, typename... Args>
  struct AcceptImpl<MemFunType C::*, Obj, Args...>
      : std::integral_constant<bool, std::is_base_of<C, Obj>::value &&
                                         absl::is_function<MemFunType>::value> {
  };

  template <typename MemFun, typename Obj, typename... Args>
  static decltype((std::declval<Obj>().*
                   std::declval<MemFun>())(std::declval<Args>()...))
  Invoke(MemFun&& mem_fun, Obj&& obj, Args&&... args) {
    return (std::forward<Obj>(obj).*
            std::forward<MemFun>(mem_fun))(std::forward<Args>(args)...);
  }
};

// ((*t1).*f)(t2, ..., tN) when f is a pointer to a member function of a
// class T and t1 is not one of the types described in the previous item.
struct MemFunAndPtr : StrippedAccept<MemFunAndPtr> {
  template <typename... Args>
  struct AcceptImpl : std::false_type {};

  template <typename MemFunType, typename C, typename Ptr, typename... Args>
  struct AcceptImpl<MemFunType C::*, Ptr, Args...>
      : std::integral_constant<bool, !std::is_base_of<C, Ptr>::value &&
                                         absl::is_function<MemFunType>::value> {
  };

  template <typename MemFun, typename Ptr, typename... Args>
  static decltype(((*std::declval<Ptr>()).*
                   std::declval<MemFun>())(std::declval<Args>()...))
  Invoke(MemFun&& mem_fun, Ptr&& ptr, Args&&... args) {
    return ((*std::forward<Ptr>(ptr)).*
            std::forward<MemFun>(mem_fun))(std::forward<Args>(args)...);
  }
};

// t1.*f when N == 1 and f is a pointer to member data of a class T and t1 is
// an object of type T or a reference to an object of type T or a reference
// to an object of a type derived from T.
struct DataMemAndRef : StrippedAccept<DataMemAndRef> {
  template <typename... Args>
  struct AcceptImpl : std::false_type {};

  template <typename R, typename C, typename Obj>
  struct AcceptImpl<R C::*, Obj>
      : std::integral_constant<bool, std::is_base_of<C, Obj>::value &&
                                         !absl::is_function<R>::value> {};

  template <typename DataMem, typename Ref>
  static decltype(std::declval<Ref>().*std::declval<DataMem>()) Invoke(
      DataMem&& data_mem, Ref&& ref) {
    return std::forward<Ref>(ref).*std::forward<DataMem>(data_mem);
  }
};

// (*t1).*f when N == 1 and f is a pointer to member data of a class T and t1
// is not one of the types described in the previous item.
struct DataMemAndPtr : StrippedAccept<DataMemAndPtr> {
  template <typename... Args>
  struct AcceptImpl : std::false_type {};

  template <typename R, typename C, typename Ptr>
  struct AcceptImpl<R C::*, Ptr>
      : std::integral_constant<bool, !std::is_base_of<C, Ptr>::value &&
                                         !absl::is_function<R>::value> {};

  template <typename DataMem, typename Ptr>
  static decltype((*std::declval<Ptr>()).*std::declval<DataMem>()) Invoke(
      DataMem&& data_mem, Ptr&& ptr) {
    return (*std::forward<Ptr>(ptr)).*std::forward<DataMem>(data_mem);
  }
};

// f(t1, t2, ..., tN) in all other cases.
struct Callable {
  // Callable doesn't have Accept because it's the last clause that gets picked
  // when none of the previous clauses are applicable.
  template <typename F, typename... Args>
  static decltype(std::declval<F>()(std::declval<Args>()...)) Invoke(
      F&& f, Args&&... args) {
    return std::forward<F>(f)(std::forward<Args>(args)...);
  }
};

// Resolves to the first matching clause.
template <typename... Args>
struct Invoker {
  typedef typename std::conditional<
      MemFunAndRef::Accept<Args...>::value, MemFunAndRef,
      typename std::conditional<
          MemFunAndPtr::Accept<Args...>::value, MemFunAndPtr,
          typename std::conditional<
              DataMemAndRef::Accept<Args...>::value, DataMemAndRef,
              typename std::conditional<DataMemAndPtr::Accept<Args...>::value,
                                        DataMemAndPtr, Callable>::type>::type>::
          type>::type type;
};

// The result type of Invoke<F, Args...>.
template <typename F, typename... Args>
using InvokeT = decltype(Invoker<F, Args...>::type::Invoke(
    std::declval<F>(), std::declval<Args>()...));

// Invoke(f, args...) is an implementation of INVOKE(f, args...) from section
// [func.require] of the C++ standard.
template <typename F, typename... Args>
InvokeT<F, Args...> Invoke(F&& f, Args&&... args) {
  return Invoker<F, Args...>::type::Invoke(std::forward<F>(f),
                                           std::forward<Args>(args)...);
}

}  // namespace base_internal
OTABSL_NAMESPACE_END
}  // namespace absl

#endif  // OTABSL_BASE_INTERNAL_INVOKE_H_

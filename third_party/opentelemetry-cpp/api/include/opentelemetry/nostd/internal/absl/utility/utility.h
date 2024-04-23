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
// This header file contains C++11 versions of standard <utility> header
// abstractions available within C++14 and C++17, and are designed to be drop-in
// replacement for code compliant with C++14 and C++17.
//
// The following abstractions are defined:
//
//   * integer_sequence<T, Ints...>  == std::integer_sequence<T, Ints...>
//   * index_sequence<Ints...>       == std::index_sequence<Ints...>
//   * make_integer_sequence<T, N>   == std::make_integer_sequence<T, N>
//   * make_index_sequence<N>        == std::make_index_sequence<N>
//   * index_sequence_for<Ts...>     == std::index_sequence_for<Ts...>
//   * apply<Functor, Tuple>         == std::apply<Functor, Tuple>
//   * exchange<T>                   == std::exchange<T>
//   * make_from_tuple<T>            == std::make_from_tuple<T>
//
// This header file also provides the tag types `in_place_t`, `in_place_type_t`,
// and `in_place_index_t`, as well as the constant `in_place`, and
// `constexpr` `std::move()` and `std::forward()` implementations in C++11.
//
// References:
//
//  https://en.cppreference.com/w/cpp/utility/integer_sequence
//  https://en.cppreference.com/w/cpp/utility/apply
//  http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3658.html

#ifndef OTABSL_UTILITY_UTILITY_H_
#define OTABSL_UTILITY_UTILITY_H_

#include <cstddef>
#include <cstdlib>
#include <tuple>
#include <utility>

#include "../base/config.h"
#include "../base/internal/inline_variable.h"
#include "../base/internal/invoke.h"
#include "../meta/type_traits.h"

namespace absl {
OTABSL_NAMESPACE_BEGIN

// integer_sequence
//
// Class template representing a compile-time integer sequence. An instantiation
// of `integer_sequence<T, Ints...>` has a sequence of integers encoded in its
// type through its template arguments (which is a common need when
// working with C++11 variadic templates). `absl::integer_sequence` is designed
// to be a drop-in replacement for C++14's `std::integer_sequence`.
//
// Example:
//
//   template< class T, T... Ints >
//   void user_function(integer_sequence<T, Ints...>);
//
//   int main()
//   {
//     // user_function's `T` will be deduced to `int` and `Ints...`
//     // will be deduced to `0, 1, 2, 3, 4`.
//     user_function(make_integer_sequence<int, 5>());
//   }
template <typename T, T... Ints>
struct integer_sequence {
  using value_type = T;
  static constexpr size_t size() noexcept { return sizeof...(Ints); }
};

// index_sequence
//
// A helper template for an `integer_sequence` of `size_t`,
// `absl::index_sequence` is designed to be a drop-in replacement for C++14's
// `std::index_sequence`.
template <size_t... Ints>
using index_sequence = integer_sequence<size_t, Ints...>;

namespace utility_internal {

template <typename Seq, size_t SeqSize, size_t Rem>
struct Extend;

// Note that SeqSize == sizeof...(Ints). It's passed explicitly for efficiency.
template <typename T, T... Ints, size_t SeqSize>
struct Extend<integer_sequence<T, Ints...>, SeqSize, 0> {
  using type = integer_sequence<T, Ints..., (Ints + SeqSize)...>;
};

template <typename T, T... Ints, size_t SeqSize>
struct Extend<integer_sequence<T, Ints...>, SeqSize, 1> {
  using type = integer_sequence<T, Ints..., (Ints + SeqSize)..., 2 * SeqSize>;
};

// Recursion helper for 'make_integer_sequence<T, N>'.
// 'Gen<T, N>::type' is an alias for 'integer_sequence<T, 0, 1, ... N-1>'.
template <typename T, size_t N>
struct Gen {
  using type =
      typename Extend<typename Gen<T, N / 2>::type, N / 2, N % 2>::type;
};

template <typename T>
struct Gen<T, 0> {
  using type = integer_sequence<T>;
};

template <typename T>
struct InPlaceTypeTag {
  explicit InPlaceTypeTag() = delete;
  InPlaceTypeTag(const InPlaceTypeTag&) = delete;
  InPlaceTypeTag& operator=(const InPlaceTypeTag&) = delete;
};

template <size_t I>
struct InPlaceIndexTag {
  explicit InPlaceIndexTag() = delete;
  InPlaceIndexTag(const InPlaceIndexTag&) = delete;
  InPlaceIndexTag& operator=(const InPlaceIndexTag&) = delete;
};

}  // namespace utility_internal

// Compile-time sequences of integers

// make_integer_sequence
//
// This template alias is equivalent to
// `integer_sequence<int, 0, 1, ..., N-1>`, and is designed to be a drop-in
// replacement for C++14's `std::make_integer_sequence`.
template <typename T, T N>
using make_integer_sequence = typename utility_internal::Gen<T, N>::type;

// make_index_sequence
//
// This template alias is equivalent to `index_sequence<0, 1, ..., N-1>`,
// and is designed to be a drop-in replacement for C++14's
// `std::make_index_sequence`.
template <size_t N>
using make_index_sequence = make_integer_sequence<size_t, N>;

// index_sequence_for
//
// Converts a typename pack into an index sequence of the same length, and
// is designed to be a drop-in replacement for C++14's
// `std::index_sequence_for()`
template <typename... Ts>
using index_sequence_for = make_index_sequence<sizeof...(Ts)>;

// Tag types

#ifdef OTABSL_USES_STD_OPTIONAL

using std::in_place_t;
using std::in_place;

#else  // OTABSL_USES_STD_OPTIONAL

// in_place_t
//
// Tag type used to specify in-place construction, such as with
// `absl::optional`, designed to be a drop-in replacement for C++17's
// `std::in_place_t`.
struct in_place_t {};

OTABSL_INTERNAL_INLINE_CONSTEXPR(in_place_t, in_place, {});

#endif  // OTABSL_USES_STD_OPTIONAL

#if defined(OTABSL_USES_STD_ANY) || defined(OTABSL_USES_STD_VARIANT)
using std::in_place_type;
using std::in_place_type_t;
#else

// in_place_type_t
//
// Tag type used for in-place construction when the type to construct needs to
// be specified, such as with `absl::any`, designed to be a drop-in replacement
// for C++17's `std::in_place_type_t`.
template <typename T>
using in_place_type_t = void (*)(utility_internal::InPlaceTypeTag<T>);

template <typename T>
void in_place_type(utility_internal::InPlaceTypeTag<T>) {}
#endif  // OTABSL_USES_STD_ANY || OTABSL_USES_STD_VARIANT

#ifdef OTABSL_USES_STD_VARIANT
using std::in_place_index;
using std::in_place_index_t;
#else

// in_place_index_t
//
// Tag type used for in-place construction when the type to construct needs to
// be specified, such as with `absl::any`, designed to be a drop-in replacement
// for C++17's `std::in_place_index_t`.
template <size_t I>
using in_place_index_t = void (*)(utility_internal::InPlaceIndexTag<I>);

template <size_t I>
void in_place_index(utility_internal::InPlaceIndexTag<I>) {}
#endif  // OTABSL_USES_STD_VARIANT

// Constexpr move and forward

// move()
//
// A constexpr version of `std::move()`, designed to be a drop-in replacement
// for C++14's `std::move()`.
template <typename T>
constexpr absl::remove_reference_t<T>&& move(T&& t) noexcept {
  return static_cast<absl::remove_reference_t<T>&&>(t);
}

// forward()
//
// A constexpr version of `std::forward()`, designed to be a drop-in replacement
// for C++14's `std::forward()`.
template <typename T>
constexpr T&& forward(
    absl::remove_reference_t<T>& t) noexcept {  // NOLINT(runtime/references)
  return static_cast<T&&>(t);
}

namespace utility_internal {
// Helper method for expanding tuple into a called method.
template <typename Functor, typename Tuple, std::size_t... Indexes>
auto apply_helper(Functor&& functor, Tuple&& t, index_sequence<Indexes...>)
    -> decltype(absl::OTABSL_OPTION_INLINE_NAMESPACE_NAME::base_internal::Invoke(
        absl::forward<Functor>(functor),
        std::get<Indexes>(absl::forward<Tuple>(t))...)) {
  return absl::OTABSL_OPTION_INLINE_NAMESPACE_NAME::base_internal::Invoke(
      absl::forward<Functor>(functor),
      std::get<Indexes>(absl::forward<Tuple>(t))...);
}

}  // namespace utility_internal

// apply
//
// Invokes a Callable using elements of a tuple as its arguments.
// Each element of the tuple corresponds to an argument of the call (in order).
// Both the Callable argument and the tuple argument are perfect-forwarded.
// For member-function Callables, the first tuple element acts as the `this`
// pointer. `absl::apply` is designed to be a drop-in replacement for C++17's
// `std::apply`. Unlike C++17's `std::apply`, this is not currently `constexpr`.
//
// Example:
//
//   class Foo {
//    public:
//     void Bar(int);
//   };
//   void user_function1(int, std::string);
//   void user_function2(std::unique_ptr<Foo>);
//   auto user_lambda = [](int, int) {};
//
//   int main()
//   {
//       std::tuple<int, std::string> tuple1(42, "bar");
//       // Invokes the first user function on int, std::string.
//       absl::apply(&user_function1, tuple1);
//
//       std::tuple<std::unique_ptr<Foo>> tuple2(absl::make_unique<Foo>());
//       // Invokes the user function that takes ownership of the unique
//       // pointer.
//       absl::apply(&user_function2, std::move(tuple2));
//
//       auto foo = absl::make_unique<Foo>();
//       std::tuple<Foo*, int> tuple3(foo.get(), 42);
//       // Invokes the method Bar on foo with one argument, 42.
//       absl::apply(&Foo::Bar, tuple3);
//
//       std::tuple<int, int> tuple4(8, 9);
//       // Invokes a lambda.
//       absl::apply(user_lambda, tuple4);
//   }
template <typename Functor, typename Tuple>
auto apply(Functor&& functor, Tuple&& t)
    -> decltype(utility_internal::apply_helper(
        absl::forward<Functor>(functor), absl::forward<Tuple>(t),
        absl::make_index_sequence<std::tuple_size<
            typename std::remove_reference<Tuple>::type>::value>{})) {
  return utility_internal::apply_helper(
      absl::forward<Functor>(functor), absl::forward<Tuple>(t),
      absl::make_index_sequence<std::tuple_size<
          typename std::remove_reference<Tuple>::type>::value>{});
}

// exchange
//
// Replaces the value of `obj` with `new_value` and returns the old value of
// `obj`.  `absl::exchange` is designed to be a drop-in replacement for C++14's
// `std::exchange`.
//
// Example:
//
//   Foo& operator=(Foo&& other) {
//     ptr1_ = absl::exchange(other.ptr1_, nullptr);
//     int1_ = absl::exchange(other.int1_, -1);
//     return *this;
//   }
template <typename T, typename U = T>
T exchange(T& obj, U&& new_value) {
  T old_value = absl::move(obj);
  obj = absl::forward<U>(new_value);
  return old_value;
}

namespace utility_internal {
template <typename T, typename Tuple, size_t... I>
T make_from_tuple_impl(Tuple&& tup, absl::index_sequence<I...>) {
  return T(std::get<I>(std::forward<Tuple>(tup))...);
}
}  // namespace utility_internal

// make_from_tuple
//
// Given the template parameter type `T` and a tuple of arguments
// `std::tuple(arg0, arg1, ..., argN)` constructs an object of type `T` as if by
// calling `T(arg0, arg1, ..., argN)`.
//
// Example:
//
//   std::tuple<const char*, size_t> args("hello world", 5);
//   auto s = absl::make_from_tuple<std::string>(args);
//   assert(s == "hello");
//
template <typename T, typename Tuple>
constexpr T make_from_tuple(Tuple&& tup) {
  return utility_internal::make_from_tuple_impl<T>(
      std::forward<Tuple>(tup),
      absl::make_index_sequence<
          std::tuple_size<absl::decay_t<Tuple>>::value>{});
}

OTABSL_NAMESPACE_END
}  // namespace absl

#endif  // OTABSL_UTILITY_UTILITY_H_

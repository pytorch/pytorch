// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <ATen/Tensor.h>
#include <ATen/VmapGeneratedPlumbing.h>

// This file contains template metaprogramming things that are used for our
// batching rules.
//
// See NOTE: [vmap plumbing] for more details on why this is necessary.
// The plumbing has a bunch of metaprogramming hacks for determining the signature
// of a batching rule from the signature of the operator, many of which use the
// helper functions in this file.

namespace at {
namespace functorch {

// Metaprogramming things
template <class... Items> using typelist = c10::guts::typelist::typelist<Items...>;
template <class TypeList> using head_t = c10::guts::typelist::head_t<TypeList>;
template <class TL1, class TL2> using concat_t = c10::guts::typelist::concat_t<TL1, TL2>;
template <typename T> class debug_t;

// tail operation
template<class TypeList>
struct tail final {
    static_assert(c10::guts::false_t<TypeList>::value,
                  "In typelist::tail<T>, the T argument must be typelist<...>.");
};
template<class Head, class... Tail>
struct tail<typelist<Head, Tail...>> final {
  using type = typelist<Tail...>;
};
template<class TypeList> using tail_t = typename tail<TypeList>::type;

template <class First, class Second, class Next, class Tail>
struct IfFirstIsTensorAndSecondisBatchDimThenTailElseNext {
  using type = Next;
};
template <class Next, class Tail>
struct IfFirstIsTensorAndSecondisBatchDimThenTailElseNext<Tensor, optional<int64_t>, Next, Tail> {
  using type = Tail;
};
template <class Next, class Tail>
struct IfFirstIsTensorAndSecondisBatchDimThenTailElseNext<const Tensor&, optional<int64_t>, Next, Tail> {
  using type = Tail;
};
template <class Next, class Tail>
struct IfFirstIsTensorAndSecondisBatchDimThenTailElseNext<Tensor&, optional<int64_t>, Next, Tail> {
  using type = Tail;
};
template <class Next, class Tail>
struct IfFirstIsTensorAndSecondisBatchDimThenTailElseNext<optional<Tensor>, optional<int64_t>, Next, Tail> {
  using type = Tail;
};
template <class Next, class Tail>
struct IfFirstIsTensorAndSecondisBatchDimThenTailElseNext<const optional<Tensor>&, optional<int64_t>, Next, Tail> {
  using type = Tail;
};
template <class Next, class Tail>
struct IfFirstIsTensorAndSecondisBatchDimThenTailElseNext<optional<Tensor>&, optional<int64_t>, Next, Tail> {
  using type = Tail;
};
template <class Next, class Tail>
struct IfFirstIsTensorAndSecondisBatchDimThenTailElseNext<std::vector<Tensor>, optional<int64_t>, Next, Tail> {
  using type = Tail;
};
template <class TypeList> struct RemoveBatchDimAfterTensor {
  using first = head_t<TypeList>;
  using next = tail_t<TypeList>;
  using second = head_t<next>;
  using tail = tail_t<next>;

  using type = concat_t<
    typelist<first>,
    typename RemoveBatchDimAfterTensor<
      typename IfFirstIsTensorAndSecondisBatchDimThenTailElseNext<first, second, next, tail>::type
    >::type
  >;
};
template <class Type> struct RemoveBatchDimAfterTensor<typelist<Type>> {
  using type = typelist<Type>;
};
template <> struct RemoveBatchDimAfterTensor<typelist<>> {
  using type = typelist<>;
};
template<class TypeList> using remove_batch_dim_after_tensor_t = typename RemoveBatchDimAfterTensor<TypeList>::type;

template <typename T> struct UnpackSingleItemTuple {
  using type = T;
};
template <typename T> struct UnpackSingleItemTuple<std::tuple<T>> {
  using type = T;
};
template <typename T> using unpack_single_item_tuple_t = typename UnpackSingleItemTuple<T>::type;

template <typename Return, typename TupleArgs> struct BuildFunctionHelper;
template <typename Return, typename... Args> struct BuildFunctionHelper<Return, std::tuple<Args...>> {
  using type = Return(Args...);
};
template <typename Return, typename TL>
struct BuildFunction {
  using type = typename BuildFunctionHelper<Return, c10::guts::typelist::to_tuple_t<TL>>::type;
};
template <typename Return, typename TL> using build_function_t = typename BuildFunction<Return, TL>::type;


template <typename batch_rule_t> struct ToOperatorType {
  using batch_rule_return_type = typename c10::guts::function_traits<batch_rule_t>::return_type;
  using batch_rule_parameter_types = typename c10::guts::function_traits<batch_rule_t>::parameter_types;

  using operator_parameter_types = remove_batch_dim_after_tensor_t<batch_rule_parameter_types>;
  using operator_return_type =
    unpack_single_item_tuple_t<
      c10::guts::typelist::to_tuple_t<
        remove_batch_dim_after_tensor_t<
          c10::guts::typelist::from_tuple_t<batch_rule_return_type>>>>;

  using type = build_function_t<operator_return_type, operator_parameter_types>;
};
template <typename batch_rule_t> using to_operator_t = typename ToOperatorType<batch_rule_t>::type;

}
} // namespace at

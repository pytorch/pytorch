// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <ATen/Tensor.h>
#include <functorch/csrc/OutOfPlacePlumbing.h>

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

// TODO: get rid of these
// Do I need templates on templates now?
// template <typename func_t> struct LowerToNextLayer {};
// template <typename Return, typename... Args> struct LowerToNextLayer<Return(Args...)> {
//   // How to pass in batch_rule directly?
//   static Return apply(Args... args);
// };


//# Tensor lowerToNextLayer(
//#     std::function<std::tuple<Tensor,optional<int64_t>>(const Tensor&, optional<int64_t>)> batch_rule,
//#     const Tensor& tensor);
std::tuple<Tensor,optional<int64_t>> abs_batch_rule(const Tensor& tensor, optional<int64_t> batch_dim);

template<typename F, F Func, typename Return, typename TupleArgs> struct TORCH_API Dummy {};

template<typename F, F Func, typename Return, typename...T> struct Dummy<F, Func, Return, std::tuple<T...>> {
  static Return apply(T... args) {
    return lowerToNextLayer(abs_batch_rule, std::forward<T>(args)...);
  }
};

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


// std::tuple<Tensor,optional<int64_t>> (*kAbsBatchRule)(const Tensor& Tensor, optional<int64_t>)
//  = &abs_batch_rule;
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

template <typename F, F Func> struct TORCH_API PrimBatchRule3 {
  using func_t = to_operator_t<typename std::remove_pointer<F>::type>;
  using result_type = typename c10::guts::function_traits<func_t>::return_type;
  using parameter_types = c10::guts::typelist::to_tuple_t<typename c10::guts::function_traits<func_t>::parameter_types>;
  static auto apply = Dummy<F, Func, result_type, parameter_types>::apply;
};

template<typename Return, typename TypeList> struct TORCH_API PrimBatchRule5 {};
template<typename Return, typename... T> struct PrimBatchRule5<Return, typelist<T...>> {
  static inline Return apply(T... args) {
    return lowerToNextLayer(abs_batch_rule, std::forward<T>(args)...);
  }
};

template<typename func_t> struct PrimBatchRule6 {};
template<typename Return, typename... Args> struct PrimBatchRule6<Return (Args...)> {
  static inline Return apply(Args... args) {
    return lowerToNextLayer(abs_batch_rule, std::forward<Args>(args)...);
  }
};

// template<typename batch_rule_t, batch_rule_t BatchRule> struct PrimBatchRule7 {};
// template<typename batch_rule_t, batch_rule_t BatchRule, typename BRReturn, typename... BRArgs>
// struct PrimBatchRule7<BRReturn(*)(BRArgs...), BatchRule> {
template<typename br_t, br_t BatchRule, typename func_t> struct PrimBatchRule7 {};
template<typename br_t, br_t BatchRule, typename Return, typename... Args> struct PrimBatchRule7<
br_t, BatchRule, Return (Args...)> {
  static inline Return apply(Args... args) {
    return lowerToNextLayer<br_t, Return, Args...>(BatchRule, std::forward<Args>(args)...);
  }
};

}
} // namespace at

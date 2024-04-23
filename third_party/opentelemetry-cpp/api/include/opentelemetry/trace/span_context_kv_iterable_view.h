// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iterator>
#include <type_traits>
#include <utility>

#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/nostd/function_ref.h"
#include "opentelemetry/nostd/span.h"
#include "opentelemetry/nostd/string_view.h"
#include "opentelemetry/nostd/utility.h"
#include "opentelemetry/trace/span_context.h"
#include "opentelemetry/trace/span_context_kv_iterable.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace trace
{
// NOTE - code within `detail` namespace implements internal details, and not part
// of the public interface.
namespace detail
{
template <class T>
inline void take_span_context_kv(SpanContext, opentelemetry::common::KeyValueIterableView<T>)
{}

template <class T, nostd::enable_if_t<common::detail::is_key_value_iterable<T>::value> * = nullptr>
inline void take_span_context_kv(SpanContext, T &)
{}

inline void take_span_context_kv(
    SpanContext,
    std::initializer_list<std::pair<nostd::string_view, common::AttributeValue>>)
{}

template <class T>
auto is_span_context_kv_iterable_impl(T iterable)
    -> decltype(take_span_context_kv(std::begin(iterable)->first, std::begin(iterable)->second),
                nostd::size(iterable),
                std::true_type{});

std::false_type is_span_context_kv_iterable_impl(...);

template <class T>
struct is_span_context_kv_iterable
{
  static const bool value =
      decltype(detail::is_span_context_kv_iterable_impl(std::declval<T>()))::value;
};
}  // namespace detail

template <class T>
class SpanContextKeyValueIterableView final : public SpanContextKeyValueIterable
{
  static_assert(detail::is_span_context_kv_iterable<T>::value,
                "Must be a context/key-value iterable");

public:
  explicit SpanContextKeyValueIterableView(const T &links) noexcept : container_{&links} {}

  bool ForEachKeyValue(nostd::function_ref<bool(SpanContext, const common::KeyValueIterable &)>
                           callback) const noexcept override
  {
    auto iter = std::begin(*container_);
    auto last = std::end(*container_);
    for (; iter != last; ++iter)
    {
      if (!this->do_callback(iter->first, iter->second, callback))
      {
        return false;
      }
    }
    return true;
  }

  size_t size() const noexcept override { return nostd::size(*container_); }

private:
  const T *container_;

  bool do_callback(SpanContext span_context,
                   const common::KeyValueIterable &attributes,
                   nostd::function_ref<bool(SpanContext, const common::KeyValueIterable &)>
                       callback) const noexcept
  {
    if (!callback(span_context, attributes))
    {
      return false;
    }
    return true;
  }

  template <class U,
            nostd::enable_if_t<common::detail::is_key_value_iterable<U>::value> * = nullptr>
  bool do_callback(SpanContext span_context,
                   const U &attributes,
                   nostd::function_ref<bool(SpanContext, const common::KeyValueIterable &)>
                       callback) const noexcept
  {
    return do_callback(span_context, common::KeyValueIterableView<U>(attributes), callback);
  }

  bool do_callback(
      SpanContext span_context,
      std::initializer_list<std::pair<nostd::string_view, common::AttributeValue>> attributes,
      nostd::function_ref<bool(SpanContext, const common::KeyValueIterable &)> callback)
      const noexcept
  {
    return do_callback(span_context,
                       nostd::span<const std::pair<nostd::string_view, common::AttributeValue>>{
                           attributes.begin(), attributes.end()},
                       callback);
  }
};
}  // namespace trace
OPENTELEMETRY_END_NAMESPACE

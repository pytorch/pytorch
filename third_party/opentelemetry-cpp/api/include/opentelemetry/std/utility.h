// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <utility>

#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
// Standard Type aliases in nostd namespace
namespace nostd
{

//
// Backport of std::data
//
// See https://en.cppreference.com/w/cpp/iterator/data
//
template <class C>
auto data(C &c) noexcept(noexcept(c.data())) -> decltype(c.data())
{
  return c.data();
}

template <class C>
auto data(const C &c) noexcept(noexcept(c.data())) -> decltype(c.data())
{
  return c.data();
}

template <class T, std::size_t N>
T *data(T (&array)[N]) noexcept
{
  return array;
}

template <class E>
const E *data(std::initializer_list<E> list) noexcept
{
  return list.begin();
}

//
// Backport of std::size
//
// See https://en.cppreference.com/w/cpp/iterator/size
//
template <class C>
auto size(const C &c) noexcept(noexcept(c.size())) -> decltype(c.size())
{
  return c.size();
}

template <class T, std::size_t N>
std::size_t size(T (&/* array */)[N]) noexcept
{
  return N;
}

template <std::size_t N>
using make_index_sequence = std::make_index_sequence<N>;

template <std::size_t... Ints>
using index_sequence = std::index_sequence<Ints...>;

}  // namespace nostd
OPENTELEMETRY_END_NAMESPACE

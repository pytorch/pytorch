#pragma once
#include <c10/core/DispatchKey.h>

namespace at {
namespace functorch {

constexpr auto kBatchedKey = c10::DispatchKey::BatchedOutOfTree;

}} // namespace at::functorch

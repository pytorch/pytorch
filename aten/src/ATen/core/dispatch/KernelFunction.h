#pragma once

#include <ATen/core/dispatch/KernelCache.h>
#include <ATen/core/stack.h>

namespace c10 {

using Stack = torch::jit::Stack; // TODO Instead of this, move torch::jit::Stack to the c10 namespace.

/**
 * This is the basic ABI for any kernel call. Each kernel is registered as a
 * pointer to a global C function of this type.
 */
using KernelFunction = void(Stack*, KernelCache* cache);

}

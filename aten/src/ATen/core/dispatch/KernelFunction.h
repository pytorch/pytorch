#pragma once

#include <ATen/core/dispatch/KernelCache.h>
#include <vector>
#include <functional>
#include <memory>

namespace c10 {

class IValue;
using Stack = std::vector<IValue>;  // TODO Instead of this, move torch::jit::Stack to the c10 namespace.

/**
 * This is the basic ABI for any kernel call. Each kernel is registered as a
 * function pointer `KernelFunction*`, i.e. kernels are not allowed to be closures.
 */
using KernelFunction = void(Stack*, KernelCache* cache);

/**
 * The type of a user-supplied function to initialize the kernel cache.
 * this is stored together with the KernelFunction in the DispatchTable
 * so we can create a new cache instance when a kernel is looked up
 * from the dispatch table.
 */
using KernelCacheCreatorFunction = std::function<std::unique_ptr<c10::KernelCache> ()>;

}

#pragma once

#include <c10/core/SymNodeImpl.h>

#include <mutex>

// The Python side of native nodes, defined in python_symbolic.cpp.

namespace torch::symbolic {

class NativeShapeEnv;
class NativeSymNodeImpl;

// The Python SymNode that `node` stands for, as a PythonSymNodeImpl. A new one
// on every call: outside proxy mode a Python SymNode op does not depend on the
// identity of its operands.
c10::SymNode materialize(const NativeSymNodeImpl& node);

// Locks env.mutex(). A thread holding the GIL releases it while it blocks, so
// no thread waits for the mutex while holding the GIL, and a thread holding
// the mutex may take the GIL (to call Python) without deadlock.
std::unique_lock<std::mutex> lock_env(NativeShapeEnv& env);

// Whether the config that ShapeEnv reads on every evaluation
// (backed_size_oblivious, aggressive_guard_free_semantics) is at its default,
// which a native env requires. Takes the GIL.
bool native_config_is_default();

} // namespace torch::symbolic

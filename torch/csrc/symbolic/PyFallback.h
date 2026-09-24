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

// Called when env.live_nodes() changes between 0 and nonzero; holds or
// releases the Python ShapeEnv to match. Takes the GIL.
void live_nodes_changed(NativeShapeEnv& env);

// Locks env.mutex(). A thread holding the GIL releases it while it blocks, so
// no thread waits for the mutex while holding the GIL, and a thread holding
// the mutex may take the GIL (to call Python) without deadlock.
std::unique_lock<std::mutex> lock_env(NativeShapeEnv& env);

// Whether the config that ShapeEnv reads on every evaluation
// (backed_size_oblivious, aggressive_guard_free_semantics) is at its default,
// which a native env requires. Takes the GIL.
bool native_config_is_default();

// Whether get_proxy_mode() is set. Does not take the GIL.
bool proxy_mode();

// SymNode.<method>(*args), the Python impl. With a native args[0] and a proxy
// mode active, it takes its proxy branch, which only touches the hints,
// wrap_node and to_node of its arguments, so the native nodes themselves reach
// handle_sym_dispatch and their proxy slots are found. With materialized
// arguments it is the fallback for methods that are not SymNodeImpl virtuals.
c10::SymNode python_impl(const char* method, c10::ArrayRef<c10::SymNode> args);

// The proxy branch of SymNode.<method>(*args) for native args whose hints the
// branch can combine without raising:
//   to_node(args[0], handle_sym_dispatch(op, tuple(map(wrap_node, args)), {}))
// with op = METHOD_TO_OPERATOR[op_name], minus the Python frames around it.
// Calls python_impl when capture_provenance would log.
c10::SymNode proxy_dispatch(
    const char* method,
    const char* op_name,
    c10::ArrayRef<c10::SymNode> args);
c10::SymNode python_impl(
    const char* method,
    const c10::SymNode& self,
    c10::ArrayRef<c10::SymNode> sizes,
    c10::ArrayRef<c10::SymNode> strides);
c10::SymNode python_impl(
    const char* method,
    const c10::SymNode& self,
    c10::ArrayRef<c10::SymNode> args);

} // namespace torch::symbolic

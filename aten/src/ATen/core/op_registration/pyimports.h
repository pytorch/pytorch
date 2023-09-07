#pragma once
#include <mutex>
#include <string>
#include <vector>

#include <c10/macros/Export.h>

namespace c10 {
namespace impl {

// NOTE: [Requiring Python imports with TORCH_LIBRARY]
//
// Motivation
// ----------
// It is possible to add kernels for operators from both Python and C++
// using the torch.library and TORCH_LIBRARY APIs. The typical case is
// adding CPU/CUDA kernels from C++ (so that they may be used with AOTInductor)
// and meta kernels / autograd formulas / other things from Python
// (because Python offers a superior authoring experience and avoids
// many pitfalls like needing to worry about symints).
//
// When someone does torch.ops.load_library("some_library.so"), we want this
// to (1) load the C++ library and run the static initializers
// AND (2) also import the python file(s) that add the additional kernels
// so that the operator is not in a half-usable state.
//
// How to use the mechanism
// ------------------------
// The API is:
// - In C++, call `m.requires_import("fbgemm.custom_operators")` inside
//   of a TORCH_LIBRARY block.
// - In the fbgemm.custom_operators module in Python, register the additional
//   kernels.
//
// Implementation details
// ----------------------
// `register_required_pyimport` appends the string to a global list, from
// which we can read from. The idea is that torch.ops.load_library should
// load the library, then read the list of strings, run imports, and then
// finally clear the list.

TORCH_API void register_required_pyimport(std::string str);
// This is not thread-safe. It should only be called from Python.
TORCH_API const std::vector<std::string>& unsafe_get_required_pyimports();
TORCH_API void clear_required_pyimports();

}
}

#pragma once

// ${generated_comment}

#include <ATen/Functions.h>
#include <ATen/Tensor.h>

// Extension writers: do you write wrapper functions? Are you frustrated with
// resolving overloads of operators? Are you frustrated with dealing with
// pointer-to-methods and resolving overloads of pointer-to-methods?? Look no
// further, this is the utility for you.
//
// Given an operator schema: aten::op.overload(...
//
// Use ATEN_FN2(op, overload) to get a *function* version of the operator
// that is guaranteed to not be overloaded. This means that you can safely
// decltype(&ATEN_FN2(op, overload)) it. NB: the 2 means this macro takes 2 args.
//
// Given an operator schema without an overload name: aten::op(...
//
// Use ATEN_FN(op) to get an unambiguous *function* version of the operator.
//
// There is some interesting behavior for out= operations.
// ATEN_FN2(sin, out) gives a function that is *faithful* to the schema;
// that is, the order of arguments is exactly what it looks like in the schema.

#define ATEN_FN2(op_name, overload) at::_ops::op_name##_##overload
#define ATEN_FN(op_name) at::_ops::op_name

// WARNING: Please do not call any of the ops in the _ops namespace directly.
// Use the ATEN_FN macros. We do not guarantee stability of the naming
// scheme for the functions in at::_ops
namespace at { namespace _ops {

// NB: We are forced to special case requires_grad_. This is because all
// of the auto-generated inplace method signatures in TensorMethods.h are
// codegen'ed to return Tensor&, but requires_grad_ has a `manual_cpp_binding`
// with a different signature that returns `const Tensor&`.
//
// Eventually, the plan is to kill Tensor& from all C++ signatures and use
// const Tensor&. When that happens, we can remove this special case and just
// let the codegen handle it.
TORCH_API Tensor & requires_grad_(Tensor & self, bool requires_grad);

${declarations}

}} // namespace at::_ops

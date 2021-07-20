#pragma once

// ${generated_comment}

#include <c10/core/Scalar.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/QScheme.h>
#include <tuple>
#include <vector>

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

#define ATEN_FN2(op_name, overload) at::_ops::op_name##_##overload::call
#define ATEN_FN(op_name) at::_ops::op_name::call

// Separately, ATEN_OP(op) and ATEN_OP2(op, overload) define a class containing compile-time
// metadata about a given aten operator.
// Notable data on the class includes:
// - ATEN_OP2(add, Tensor)::name // returns the string name: "add"
// - ATEN_OP2(add, Tensor)::overload_name // returns the string overload name: "Tensor"
// - ATEN_OP2(add, Tensor)::schema // returns the C++ schema type: at::Tensor (const at::Tensor &, const at::Tensor &, const at::Scalar &)
// - ATEN_OP2(add, Tensor)::schema_str // returns the string jit type: "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"

#define ATEN_OP2(op_name, overload) at::_ops::op_name##_##overload
#define ATEN_OP(op_name) at::_ops::op_name

// WARNING: Please do not call any of the ops in the _ops namespace directly.
// Use the ATEN_FN macros. We do not guarantee stability of the naming
// scheme for the functions in at::_ops

// See Note [The ATen Operators API] for details of the at::_ops namespace
namespace c10 {

template<typename T>
class optional;
template<typename T>
class List;
class Stream;
struct Storage;

}

namespace at {

class Tensor;
struct Dimname;
struct Generator;
using TensorList = c10::ArrayRef<Tensor>;
using DimnameList = c10::ArrayRef<Dimname>;
using Stream = c10::Stream;
using Storage = c10::Storage;
using QScheme = c10::QScheme;

namespace _ops {

${declarations}

}} // namespace at::_ops

#include <ATen/core/TensorBody.h>

// TODO Undo all logic introduced for Note [Avoiding Include Cycles In Static Dispatch]
// Code introduced to avoid cyclic dependency in static dispatch is no longer
// needed as static dispatch logic is moved from TensorBody.h, which caused cycles in the first place,
// to Operators.cpp for supporting multiple backends with multiple kernels.
//
// Note [Avoiding Include Cycles In Static Dispatch]
// In order to avoid #include cycles in the static dispatch build, we've carefully split out
// the static function definition files into {DispatchKey}Functions.h and {DispatchKey}Functions_inl.h.
//
// Without this split, the include cycle looks like TensorBody.h -> CPUFunctions.h -> TensorBody.h.
// - TensorBody.h #includes CPUFunctions.h in the static dispatch build, because the tensor methods
//   all need to call into the fastpath C++ API defined in CPUFunctions.h. The methods are also all
//   directly inlined into TensorBody.h.
// - CPUFunctions.h #includes TensorBody.h because it contains function declarations for the entire C++ API,
//   which include functions that have defaultable std::optional<Tensor> arguments.
//   That requires knowing the full Tensor class definition.
//
// We break the cycle by doing the following:
// - Split out CPUFunction.h into two files: CPUFunctions.h and CPUFunctions_inl.h
// - CPUFunction.h is a dummy file that just includes the Tensor class and includes CPUFunctions_inl.,
// - CPUFunctions_inl.h includes everything else
// - (only in the static dispatch build) TensorBody.h makes sure to finish defining the Tensor class,
//   and then it includes CPUFunctions_inl.h.
// - All other files that want the cpu fastpath functions can include CPUFunctions.h directly.
// - This also means that static dispatch build, CPUFunctions.h only needs to
//   #include TensorBody.h, and it will automatically bring in CPUFunctions_inl.h.
${inline_headers}

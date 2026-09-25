#include <c10/core/SymIntArrayRef.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>

namespace c10 {

namespace {

std::string stringifyProblemAndArray(SymIntArrayRef ar, const SymInt& problem) {
  try {
    return c10::str(": ", problem, " in SymIntArrayRef ", ar);
  } catch (const std::exception&) {
    // Diagnostic enrichment must not mask the conversion error.
    return " (value and full array unavailable because stringification failed)";
  }
}

} // namespace

std::string formatSymIntArrayRefToIntArrayRefError(
    SymIntArrayRef ar,
    const SymInt& problem,
    const char* file,
    int64_t line) {
  const auto problem_index = static_cast<size_t>(&problem - ar.data());
  const bool is_symbolic = problem.is_symbolic();
  return c10::str(
      file,
      ":",
      line,
      ": SymIntArrayRef expected to contain only concrete integers that are "
      "stored inline and can be viewed as an IntArrayRef. Found ",
      is_symbolic ? "symbolic SymInt" : "heap-allocated concrete SymInt",
      " at index ",
      problem_index,
      stringifyProblemAndArray(ar, problem),
      is_symbolic
          ? ". This commonly happens when an operator/kernel does not support "
            "symbolic shapes at this dispatch key. Common causes include "
            "calling an eager factory/kernel with a symbolic shape outside "
            "FakeTensorMode, an operator/kernel missing SymInt support, or "
            "running under a Python dispatch mode without the Python "
            "dispatcher enabled. If this is expected during fake/meta "
            "tracing, make sure FakeTensorMode and the Python dispatcher are "
            "active; otherwise specialize or guard the symbolic size before "
            "this call."
          : ". This value is concrete, but the non-owning IntArrayRef "
            "conversion used here cannot represent heap-allocated SymInt "
            "values. Use an owning conversion such as asIntArrayRefSlowAlloc "
            "to materialize the values into a DimVector before this call.");
}

} // namespace c10

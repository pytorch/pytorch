#pragma once

#include <torch/csrc/python_headers.h>
#include <string>
#include <vector>

namespace torch {

// If mismatch_reasons is not empty, it holds one entry per option saying why
// the arguments do not match it (empty to say nothing), and is printed instead
// of the per-argument match that is guessed from the option strings.
std::string format_invalid_args(
    PyObject* given_args,
    PyObject* given_kwargs,
    const std::string& function_name,
    const std::vector<std::string>& options,
    const std::vector<std::string>& mismatch_reasons = {});

} // namespace torch

// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#pragma once

#include <string>
#include <vector>

namespace c10 {
struct IValue;
}

namespace torch {
namespace jit {

struct Node;
class StaticModule;

namespace test {

// Given a model/function in jit or IR script, run the model/function
// with the jit interpreter and static runtime, and compare the results
void testStaticRuntime(
    const std::string& source,
    const std::vector<c10::IValue>& args,
    const std::vector<c10::IValue>& args2 = {},
    const bool use_allclose = false,
    const bool use_equalnan = false);

} // namespace test
} // namespace jit
} // namespace torch

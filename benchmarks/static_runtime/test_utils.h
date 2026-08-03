// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <string>
#include <vector>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/static/impl.h>

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
    const bool use_equalnan = false,
    const bool check_resize = true);

std::shared_ptr<Graph> getGraphFromScript(const std::string& jit_script);

std::shared_ptr<Graph> getGraphFromIR(const std::string& ir);

bool hasProcessedNodeWithName(
    torch::jit::StaticModule& smodule,
    const char* name);

at::Tensor getTensor(const at::IValue& ival);

Node* getNodeWithKind(const StaticModule& smodule, const std::string& kind);
Node* getNodeWithKind(std::shared_ptr<Graph>& graph, const std::string& kind);

bool hasNodeWithKind(const StaticModule& smodule, const std::string& kind);
bool hasNodeWithKind(std::shared_ptr<Graph>& graph, const std::string& kind);

void compareResultsWithJIT(
    StaticRuntime& runtime,
    const std::shared_ptr<Graph>& graph,
    const std::vector<c10::IValue>& args,
    const bool use_allclose = false,
    const bool use_equalnan = false);

void compareResults(
    const IValue& expect,
    const IValue& actual,
    const bool use_allclose = false,
    const bool use_equalnan = false);

} // namespace test
} // namespace jit
} // namespace torch

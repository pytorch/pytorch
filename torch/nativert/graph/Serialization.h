#pragma once

#include <torch/nativert/graph/Graph.h>

#include <torch/csrc/utils/generated_serialization_types.h>

namespace torch::nativert {
/**
 * This file contains serialization utilities for Graph.
 *
 * There are two serialized representations we care about:
 * - Json: stable but hard to work with, not really human readable
 * - Debug format: human-readable, not stable.
 */

// Json -> Graph
std::unique_ptr<Graph> jsonToGraph(
    const torch::_export::GraphModule& jsonGraph,
    bool loadNodeMetadata = true);

bool isSymbolic(const torch::_export::Argument& arg);

Constant constantToValue(
    const torch::_export::Argument& jsonArg,
    bool loadNodeMetadata);

} // namespace torch::nativert

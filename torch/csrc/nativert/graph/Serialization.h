#pragma once

#include "torch/csrc/nativert/graph/Graph.h"

#include "torch/csrc/utils/generated_serialization_types.h" // @manual=//caffe2:torch-cpp-cpu

namespace torch::nativert {
/**
 * This file contains serialization utilities for Graph.
 *
 * There are two serialized representations we care about:
 * - Json: stable but hard to work with, not really human readable
 * - Debug format: human-readable, not stable.
 *
 * All formats should be logically equivalent, so we should be able to go from
 * in-memory graph <> json <> debugformat interchangeably
 */

// Json -> Graph
std::unique_ptr<Graph> jsonToGraph(
    const torch::_export::GraphModule& jsonGraph,
    bool loadNodeMetadata = true);

// Graph -> Json
std::pair<torch::_export::Graph, torch::_export::GraphSignature> graphToJson(
    const Graph& graph);

bool isSymbolic(const torch::_export::Argument& arg);

Constant constantToValue(
    const torch::_export::Argument& jsonArg,
    bool loadNodeMetadata);

} // namespace torch::nativert

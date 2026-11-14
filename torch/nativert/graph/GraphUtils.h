#pragma once

namespace torch::nativert {

class Node;

/**
 * Utility functions for working with Graph nodes and values.
 */

/**
 * Check if all input/output tensors are on CPU and all device-type attributes
 * have the value of 'cpu'. This is a util function to check if a Node can use
 * static dispatch CPU kernels.
 *
 * @param node The node to check
 * @return true if all I/O tensors and device attributes are on CPU, false
 * otherwise
 */
bool areAllIOTensorsAttributesOnCpu(const Node& node);

} // namespace torch::nativert

#pragma once

namespace torch::_export {

/// Register example upgraders for the upgrader system for testing.
/// This function demonstrates common upgrade patterns and is primarily
/// used for testing and demonstration purposes.
void registerExampleUpgraders();

/// Deregister example upgraders for the upgrader system for testing.
/// This function cleans up the example upgraders that were registered
/// by registerExampleUpgraders().
void deregisterExampleUpgraders();

} // namespace torch::_export

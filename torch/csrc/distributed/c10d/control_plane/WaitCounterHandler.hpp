#pragma once

#include <string>

namespace c10d {
namespace control_plane {

// Returns all wait counter values as a JSON string
std::string getWaitCounterValuesJson();

// Ensures the wait counter backend is registered
void ensureWaitCounterBackendRegistered();

} // namespace control_plane
} // namespace c10d

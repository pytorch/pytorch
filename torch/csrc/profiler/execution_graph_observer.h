#pragma once

#include <string>

namespace torch {
namespace profiler {
namespace impl {

// Adds the execution graph observer as a global callback function, the data
// will be written to output file path.
bool addExecutionGraphObserver(const std::string& output_file_path);

// Remove the execution graph observer from the global callback functions.
void removeExecutionGraphObserver();

// Enables execution graph observer.
void enableExecutionGraphObserver();

// Disables execution graph observer.
void disableExecutionGraphObserver();

} // namespace impl
} // namespace profiler
} // namespace torch

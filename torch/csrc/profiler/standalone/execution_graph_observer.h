#pragma once

#include <c10/macros/Export.h>
#include <string>

namespace torch {
namespace profiler {
namespace impl {

// Adds the execution graph observer as a global callback function, the data
// will be written to output file path.
TORCH_API bool addExecutionGraphObserver(const std::string& output_file_path);

// Remove the execution graph observer from the global callback functions.
TORCH_API void removeExecutionGraphObserver();

// Enables execution graph observer.
TORCH_API void enableExecutionGraphObserver();

// Disables execution graph observer.
TORCH_API void disableExecutionGraphObserver();

} // namespace impl
} // namespace profiler
} // namespace torch

#pragma once

/**
 * @file
 * Declarations shared between import_data.cpp and export_data.cpp
 */

namespace torch {
namespace jit {
namespace mobile {

#if defined(ENABLE_FLATBUFFER)

namespace internal {
/**
 * The name of the mobile::Module attribute which contains saved parameters, as
 * a Dict of names to Tensors. Only used for Flatbuffer serialization.
 */
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
constexpr char kSavedParametersAttributeName[] = "data";
} // namespace internal

#endif // defined(ENABLE_FLATBUFFER)

} // namespace mobile
} // namespace jit
} // namespace torch

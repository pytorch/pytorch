#pragma once

#include <string>

namespace torch {
namespace jit {

// Utility functions to maintain consistency between import and export paths.
namespace ImportExportHelpers {

// Convert a class type's qualifier name to the corresponding path the source
// file it should be written to.
//
// Qualifier is like: foo.bar.baz
// Returns: libs/foo/bar/baz.py
std::string qualifierToPath(
    const std::string& qualifier,
    const std::string& export_prefix);

// Convert a source file path to a class type's qualifier name.
//
// Path is like: libs/foo/bar/baz.py
// Returns: foo.bar.baz
std::string pathToQualifier(
    const std::string& classPath,
    const std::string& export_prefix);

} // namespace ImportExportHelpers
} // namespace jit
} // namespace torch

#pragma once
#include <cstddef>

namespace torch {
namespace jit {
constexpr size_t BYTECODE_INDEX_INSTRUCTION = 0;
constexpr size_t BYTECODE_INDEX_OPERATOR = 1;
constexpr size_t BYTECODE_INDEX_CONSTANT = 2;
constexpr size_t BYTECODE_INDEX_TYPE = 3;
constexpr size_t BYTECODE_INDEX_REGISTER_SIZE = 4;

constexpr size_t BYTECODE_INDEX_SCHEMA_ARGUMENTS = 0;
constexpr size_t BYTECODE_INDEX_SCHEMA_RETURNS = 1;

constexpr size_t BYTECODE_INDEX_ARGUMENT_NAME = 0;
constexpr size_t BYTECODE_INDEX_ARGUMENT_TYPE = 1;
constexpr size_t BYTECODE_INDEX_ARGUMENT_DEFAULT_VALUE = 2;

constexpr size_t BYTECODE_INDEX_MODULE_DEBUG_HANDLES = 0;
} // namespace jit
} // namespace torch

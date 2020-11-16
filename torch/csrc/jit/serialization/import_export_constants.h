#pragma once

namespace torch {
namespace jit {
constexpr size_t BYTECODE_INDEX_INSTRUCTION = 0;
constexpr size_t BYTECODE_INDEX_OPERATOR = 1;
constexpr size_t BYTECODE_INDEX_CONSTANT = 2;
constexpr size_t BYTECODE_INDEX_TYPE = 3;
constexpr size_t BYTECODE_INDEX_REGISTER_SIZE = 4;
constexpr size_t BYTECODE_INDEX_MODULE_DEBUG_INFO = 0;
} // namespace jit
} // namespace torch

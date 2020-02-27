
copy: fbcode/caffe2/torch/csrc/jit/frontend/parser_constants.h
copyrev: a0b7cedf3007a28ecc21294d2e611ea650961f1f

#pragma once

namespace torch {
namespace jit {
namespace script {
static const char* valid_single_char_tokens = "+-*/%@()[]:,={}><.?!&^|~";
} // namespace script
} // namespace jit
} // namespace torch

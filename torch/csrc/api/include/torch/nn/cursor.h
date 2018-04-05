#pragma once

namespace torch { namespace nn {
// "Cursor"s allow recursive traversal of the module tree with a depth-first or
// breadth-first policy. They have `begin()` and `end()` and can be used like
// containers for iteration
struct ModuleCursor {};
struct ParameterCursor {};
struct BufferCursor {};

}} // namespace torch::nn

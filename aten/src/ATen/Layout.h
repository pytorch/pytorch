#pragma once

namespace at {
struct Type;
enum class Backend;
} // namespace at

namespace at {
enum class Layout { Strided, Sparse };

constexpr auto kStrided = Layout::Strided;
constexpr auto kSparse = Layout::Sparse;

Layout layout_from_type(const Type& type);
Layout layout_from_backend(Backend backend);
} // namespace at

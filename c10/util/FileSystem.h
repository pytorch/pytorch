// Shim header for filesystem for compilers that are too old to have it not
// in the experimental namespace

#if __has_include(<filesystem>)
#include <filesystem>
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
#else
#error "Neither <filesystem> nor <experimental/filesystem> is available."
#endif

namespace c10 {

#if __has_include(<filesystem>)
// NOLINTNEXTLINE(misc-unused-alias-decls)
namespace filesystem = std::filesystem;
#elif __has_include(<experimental/filesystem>)
// NOLINTNEXTLINE(misc-unused-alias-decls)
namespace filesystem = std::experimental::filesystem;
#endif

} // namespace c10

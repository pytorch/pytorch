#include <regex>

namespace c10 {
namespace cuda {
namespace CUDACachingAllocator {
namespace {

class CachingAllocatorConfig {
 public:
  static size_t max_split_size() {
    return instance().m_max_split_size;
  }

  static const std::string& allocator_backend() {
    return instance().m_allocator_backend;
  }

 private:
  static std::once_flag s_flag;
  static CachingAllocatorConfig* s_instance;
  static CachingAllocatorConfig& instance() {
    std::call_once(s_flag, &CachingAllocatorConfig::init);
    return *s_instance;
  }
  static void init() {
    s_instance = new CachingAllocatorConfig();
    s_instance->parseArgs();
  }

  CachingAllocatorConfig()
      : m_max_split_size(std::numeric_limits<size_t>::max()),
        m_allocator_backend("native") {}
  size_t m_max_split_size;
  std::string m_allocator_backend;

  void parseArgs() {
    const char* val = getenv("PYTORCH_CUDA_ALLOC_CONF");
    if (val != NULL) {
      const std::string config(val);

      std::regex exp("[\\s,]+");
      std::sregex_token_iterator it(config.begin(), config.end(), exp, -1);
      std::sregex_token_iterator end;
      std::vector<std::string> options(it, end);

      for (auto option : options) {
        std::regex exp2("[:]+");
        std::sregex_token_iterator it2(option.begin(), option.end(), exp2, -1);
        std::sregex_token_iterator end2;
        std::vector<std::string> kv(it2, end2);
        if (kv.size() >= 2) {
          /* Maximum split size in MB.  Limited to large size blocks */
          if (kv[0].compare("max_split_size_mb") == 0) {
            size_t val2 = stoi(kv[1]);
            TORCH_CHECK(
                val2 > kLargeBuffer / (1024 * 1024),
                "CachingAllocator option max_split_size_mb too small, must be >= ",
                kLargeBuffer / (1024 * 1024),
                "");
            val2 = std::max(val2, kLargeBuffer / (1024 * 1024));
            val2 = std::min(
                val2, (std::numeric_limits<size_t>::max() / (1024 * 1024)));
            m_max_split_size = val2 * 1024 * 1024;
          } else if (kv[0].compare("backend") == 0) {
            TORCH_CHECK(((kv[1].compare("native") == 0) ||
                         (kv[1].compare("cudaMallocAsync") == 0)),
                        "Unknown allocator backend, "
                        "options are native and cudaMallocAsync");
            m_allocator_backend = kv[1];
          } else {
            TORCH_CHECK(false, "Unrecognized CachingAllocator option: ", kv[0]);
          }
        }
      }
    }
  }
};
CachingAllocatorConfig* CachingAllocatorConfig::s_instance;
std::once_flag CachingAllocatorConfig::s_flag;

} // anonymous namespace

// We could probably use static initializers in the interface functions here
// to eliminate need for the singleton above, but why bother, it works as-is
// and these calls are rare.

// Public interface
const std::string& allocatorBacked() {
  return CachingAllocatorConfig::allocator_backend();
}

size_t maxSplitSize() {
  return CachingAllocatorConfig::max_split_size()
}

} // namespace CUDACachingAllocator
} // namespace cuda
} // namespace c10

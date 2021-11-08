#include <regex>

namespace c10 {
namespace cuda {
namespace CUDACachingAllocator {
namespace {

size_t m_max_split_size = std::numeric_limits<size_t>::max();
std::string m_allocator_backend = "native"

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

} // anonymous namespace

// Public interface
const std::string& allocatorBacked() {
  static const std::string backend = []() {
                                       parseArgs();
                                       return m_allocator_backend;
                                     }();
  return backend;
}

size_t maxSplitSize() {
  static const size_t size = []() {
                               parseArgs();
                               return m_max_split_size;
                             }();
  return size;
}

} // namespace CUDACachingAllocator
} // namespace cuda
} // namespace c10

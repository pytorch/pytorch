#include <test/test_utils.h>

#include <c10/util/Exception.h>

#include <ops/arith.h>

#include <sstream>
#include <string_view>

namespace nvfuser {

int64_t prime_number(int64_t i) {
  static std::vector<int64_t> p{
      2,    3,    5,    7,    11,   13,   17,   19,   23,   29,   31,   37,
      41,   43,   47,   53,   59,   61,   67,   71,   73,   79,   83,   89,
      97,   101,  103,  107,  109,  113,  127,  131,  137,  139,  149,  151,
      157,  163,  167,  173,  179,  181,  191,  193,  197,  199,  211,  223,
      227,  229,  233,  239,  241,  251,  257,  263,  269,  271,  277,  281,
      283,  293,  307,  311,  313,  317,  331,  337,  347,  349,  353,  359,
      367,  373,  379,  383,  389,  397,  401,  409,  419,  421,  431,  433,
      439,  443,  449,  457,  461,  463,  467,  479,  487,  491,  499,  503,
      509,  521,  523,  541,  547,  557,  563,  569,  571,  577,  587,  593,
      599,  601,  607,  613,  617,  619,  631,  641,  643,  647,  653,  659,
      661,  673,  677,  683,  691,  701,  709,  719,  727,  733,  739,  743,
      751,  757,  761,  769,  773,  787,  797,  809,  811,  821,  823,  827,
      829,  839,  853,  857,  859,  863,  877,  881,  883,  887,  907,  911,
      919,  929,  937,  941,  947,  953,  967,  971,  977,  983,  991,  997,
      1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069,
      1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163,
      1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223};
  return p.at(i);
}

void assertCUDAKernel(Fusion* fusion, const std::string& expected_kernel) {
  const std::string actual_kernel =
      "\n" + codegen::generateCudaKernel(GpuLower(fusion).kernel());
  if (expected_kernel.size() != actual_kernel.size() ||
      expected_kernel.compare(actual_kernel) != 0) {
    std::cerr
        << " Codegen mismatch, codegen possibly changed, or is incorrect. "
        << " \n ========= EXPECTED ========= \n"
        << expected_kernel << "\n========= ACTUAL ========== \n"
        << actual_kernel << "\n=================" << std::endl;
    auto it = std::mismatch(
        expected_kernel.begin(),
        expected_kernel.end(),
        actual_kernel.begin(),
        actual_kernel.end());
    std::string actual_mismatched_snippet(it.second, actual_kernel.end());
    actual_mismatched_snippet = actual_mismatched_snippet.substr(0, 10);
    std::string expected_mismatched_snippet(it.first, expected_kernel.end());
    expected_mismatched_snippet = expected_mismatched_snippet.substr(0, 10);
    std::cerr << "First mismatch found at: " << actual_mismatched_snippet
              << ", expected: " << expected_mismatched_snippet << std::endl;
    TORCH_CHECK(false);
  }
}

namespace sass {

// For SASS instruction definitions, see:
// https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-reference
//
// Some useful informations for Ampere/Ada:
//
// Instruction format:
//   (instruction) (destination) (source1), (source2) ...
//
// Registers:
// - RX for registers
// - URX for uniform registers
// - SRX for special system-controlled registers
// - PX for predicate registers
// - UPX for uniform predicate registers
// - c[X][Y] for constant memory

namespace {

// trim: remove spaces before and after the string view
// implementation borrowed from https://stackoverflow.com/a/17976541
inline std::string_view trim(const std::string_view& s) {
  auto wsfront = std::find_if_not(
      s.begin(), s.end(), [](int c) { return std::isspace(c); });
  auto wsback = std::find_if_not(s.rbegin(), s.rend(), [](int c) {
                  return std::isspace(c);
                }).base();
  return (
      wsback <= wsfront ? "" : s.substr(wsfront - s.begin(), wsback - wsfront));
}

// Copied from LLVM libcxx C++20 implementation of
// basic_string_view::starts_with
// https://github.com/llvm/llvm-project/blob/11d8f726d24d90c67e0e99aa8e9de48d17adb750/libcxx/include/string_view#L696-L697
bool starts_with(std::string_view self, std::string_view __s) noexcept {
  return self.size() >= __s.size() && self.compare(0, __s.size(), __s) == 0;
}

} // namespace

std::string Instruction::predicate() {
  if (str[0] == '@') {
    std::stringstream ss(str);
    char ignore_at;
    std::string result;
    ss >> ignore_at >> result;
    return result;
  }
  return {};
}

std::string Instruction::action() {
  std::string result;
  std::stringstream ss(str);
  if (str[0] == '@') {
    ss >> result;
  }
  std::getline(ss, result);
  result = trim(result);
  return result;
}

std::string Instruction::op() {
  std::stringstream ss(action());
  std::string result;
  ss >> result;
  return result;
}

std::string Instruction::opCode() {
  std::string result;
  for (auto i : op()) {
    if (i == '.') {
      return result;
    }
    result.push_back(i);
  }
  return result;
}

std::vector<std::string> Instruction::args() {
  std::stringstream ss(action());
  std::string all_args;
  ss >> all_args; // discard
  std::getline(ss, all_args);
  all_args = trim(all_args);
  std::vector<std::string> result;

  std::string_view args_view(all_args);
  while (!args_view.empty()) {
    auto comma_pos = args_view.find_first_of(',');
    auto token = args_view.substr(0, comma_pos);
    token = trim(token);
    result.push_back(std::string(token));

    args_view = (comma_pos != std::string_view::npos)
        ? args_view.substr(comma_pos + 1)
        : "";
  }
  return result;
}

std::vector<std::string> Instruction::modifiers() {
  std::vector<std::string> result;
  std::string current;
  bool found_opcode = false;
  for (auto i : op()) {
    if (i == '.') {
      if (found_opcode) {
        result.push_back(current);
      }
      found_opcode = true;
      current.clear();
      continue;
    }
    current.push_back(i);
  }
  if (found_opcode) {
    result.push_back(current);
  }
  return result;
}

std::string Container::toString() {
  std::stringstream ss;
  for (auto& [key, value] : attributes) {
    ss << "." << key << ":\t" << value << std::endl;
  }
  for (auto& i : code) {
    std::visit(
        [&ss](auto&& i) {
          using T = std::decay_t<decltype(i)>;
          if constexpr (std::is_same_v<Instruction, T>) {
            ss << i.str << std::endl;
          } else if constexpr (std::is_same_v<T, Label>) {
            ss << "." << i.name << ":" << std::endl;
          }
        },
        i);
  }
  return ss.str();
}

Container parse(const std::string& nvdisasm_output) {
  Container result;
  bool started = false;
  std::stringstream ss(nvdisasm_output);
  std::string header;
  for (std::string line; std::getline(ss, line);) {
    line = trim(line);
    if (line.empty() || starts_with(line, "//")) {
      continue;
    }
    if (started) {
      if (line[0] == '.') {
        std::stringstream ss(line);
        Label l;
        char ignore_dot;
        ss >> ignore_dot >> l.name;
        l.name.resize(l.name.size() - 1); // remove trailing :
        result.code.push_back(l);
      } else {
        Instruction i;
        std::stringstream ss(line);
        char ignore;
        // parse /*address*/
        ss >> ignore >> ignore >> std::hex >> i.address >> ignore >> ignore;
        std::getline(ss, i.str);
        i.str = trim(i.str);
        i.str.resize(i.str.size() - 1); // remove trailing ;
        i.str = trim(i.str);
        result.code.push_back(i);
      }
    } else {
      if (line == header) {
        started = true;
      } else if (line[0] == '.') {
        std::stringstream ss(line);
        std::string key, value;
        char ignore;
        ss >> ignore >> key >> value;
        result.attributes[key] = value;
        if (key == "global") {
          header = ".text." + value + ":";
        }
      }
    }
  }
  return result;
}

} // namespace sass

TensorView* matmul(TensorView* a, TensorView* b, MatmulLayout layout) {
  TORCH_CHECK(
      a->nDims() == 2 && b->nDims() == 2, "only pure matmuls for these tests");
  TensorView *tv2 = nullptr, *tv0b = nullptr, *tv1b = nullptr;
  switch (layout) {
    case MatmulLayout::TT:
      tv0b = broadcast(a, {false, false, true});
      tv1b = broadcast(b, {true, false, false});
      tv2 = fusedMultiplySum(tv0b, tv1b, {1});
      break;
    case MatmulLayout::TN:
      tv0b = broadcast(a, {false, true, false});
      tv1b = broadcast(b, {true, false, false});
      tv2 = fusedMultiplySum(tv0b, tv1b, {2});
      break;
    case MatmulLayout::NT:
      tv0b = broadcast(a, {false, false, true});
      tv1b = broadcast(b, {false, true, false});
      tv2 = fusedMultiplySum(tv0b, tv1b, {0});
      break;
    default:
      TORCH_CHECK(false, "unsupported data layout.");
  }
  return tv2;
}

at::Tensor atMatmul(at::Tensor a, at::Tensor b, MatmulLayout layout) {
  switch (layout) {
    case MatmulLayout::TT:
      return a.matmul(b);
    case MatmulLayout::TN:
      return a.matmul(b.t());
    case MatmulLayout::NT:
      return a.t().matmul(b);
    default:
      TORCH_CHECK(false, "unsupported data layout.");
  }
  return at::Tensor();
}

std::pair<at::Tensor, at::Tensor> fp16MatmulAtInput(
    int M,
    int N,
    int K,
    MatmulLayout layout) {
  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);

  switch (layout) {
    case MatmulLayout::TT:
      return std::make_pair(
          at::randn({M, K}, options), at::randn({K, N}, options));
    case MatmulLayout::TN:
      return std::make_pair(
          at::randn({M, K}, options), at::randn({N, K}, options));
    case MatmulLayout::NT:
      return std::make_pair(
          at::randn({K, M}, options), at::randn({K, N}, options));
    default:
      TORCH_CHECK(false, "unsupported data layout.");
  }
  return std::make_pair(at::Tensor(), at::Tensor());
}

} // namespace nvfuser

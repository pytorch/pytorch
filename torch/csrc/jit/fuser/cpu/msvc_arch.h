// Example code extracted from MSDN page of __cpuidex

#include <intrin.h>
#include <array>
#include <bitset>
#include <string>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cpu {

class InstructionSet {
  // forward declarations
  class InstructionSet_Internal;

 public:
  // getters

  static bool AVX(void) {
    return CPU_Rep.f_1_ECX_[28];
  }
  static bool AVX2(void) {
    return CPU_Rep.f_7_EBX_[5];
  }
  static bool AVX512F(void) {
    return CPU_Rep.f_7_EBX_[16];
  }

 private:
  static const InstructionSet_Internal CPU_Rep;

  class InstructionSet_Internal {
   public:
    InstructionSet_Internal()
        : nIds_{0},
          nExIds_{0},
          f_1_ECX_{0},
          f_1_EDX_{0},
          f_7_EBX_{0},
          f_7_ECX_{0},
          f_81_ECX_{0},
          f_81_EDX_{0},
          data_{},
          extdata_{} {
      // int cpuInfo[4] = {-1};
      std::array<int, 4> cpui;

      // Calling __cpuid with 0x0 as the function_id argument
      // gets the number of the highest valid function ID.
      __cpuid(cpui.data(), 0);
      nIds_ = cpui[0];

      for (int i = 0; i <= nIds_; ++i) {
        __cpuidex(cpui.data(), i, 0);
        data_.push_back(cpui);
      }

      // load bitset with flags for function 0x00000001
      if (nIds_ >= 1) {
        f_1_ECX_ = data_[1][2];
        f_1_EDX_ = data_[1][3];
      }

      // load bitset with flags for function 0x00000007
      if (nIds_ >= 7) {
        f_7_EBX_ = data_[7][1];
        f_7_ECX_ = data_[7][2];
      }

      // Calling __cpuid with 0x80000000 as the function_id argument
      // gets the number of the highest valid extended ID.
      __cpuid(cpui.data(), 0x80000000);
      nExIds_ = cpui[0];

      for (int i = 0x80000000; i <= nExIds_; ++i) {
        __cpuidex(cpui.data(), i, 0);
        extdata_.push_back(cpui);
      }

      // load bitset with flags for function 0x80000001
      if (nExIds_ >= 0x80000001) {
        f_81_ECX_ = extdata_[1][2];
        f_81_EDX_ = extdata_[1][3];
      }
    };

    int nIds_;
    int nExIds_;
    std::bitset<32> f_1_ECX_;
    std::bitset<32> f_1_EDX_;
    std::bitset<32> f_7_EBX_;
    std::bitset<32> f_7_ECX_;
    std::bitset<32> f_81_ECX_;
    std::bitset<32> f_81_EDX_;
    std::vector<std::array<int, 4>> data_;
    std::vector<std::array<int, 4>> extdata_;
  };
};

// Initialize static member data
const InstructionSet::InstructionSet_Internal InstructionSet::CPU_Rep;

} // namespace cpu
} // namespace fuser
} // namespace jit
} // namespace torch

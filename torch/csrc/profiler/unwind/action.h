#pragma once
#include <cstdint>
#include <ostream>

namespace torch::unwind {

enum {
  A_UNDEFINED = 0x0,
  A_REG_PLUS_DATA = 0x1, // exp = REG[reg] + data0
  A_LOAD_CFA_OFFSET = 0x2, // exp = *(cfa + data0)
  A_REG_PLUS_DATA_DEREF = 0x3 // exp = *(REG[reg] + data0)
};

// DWARF register numbers — architecture-specific
#if defined(__x86_64__)
enum {
  D_UNDEFINED = -1,
  D_RBP = 6,
  D_RSP = 7,
  D_RIP = 16,
  D_REG_SIZE = 17,
};
static constexpr int D_FRAME_PTR = D_RBP;
static constexpr int D_STACK_PTR = D_RSP;
static constexpr int D_RET_ADDR = D_RIP;
static constexpr int D_EXPECTED_RA_REG = 16;
#elif defined(__aarch64__)
enum {
  D_UNDEFINED = -1,
  D_FP = 29,
  D_LR = 30,
  D_SP = 31,
  D_REG_SIZE = 32,
};
static constexpr int D_FRAME_PTR = D_FP;
static constexpr int D_STACK_PTR = D_SP;
static constexpr int D_RET_ADDR = D_LR;
static constexpr int D_EXPECTED_RA_REG = 30;
#else
enum {
  D_UNDEFINED = -1,
  D_REG_SIZE = 1,
};
#endif

struct Action {
  uint8_t kind = A_UNDEFINED;
  int32_t reg = -1;
  int64_t data = 0;
  static Action undefined() {
    return Action{A_UNDEFINED};
  }
  static Action regPlusData(int32_t reg, int64_t offset) {
    return Action{A_REG_PLUS_DATA, reg, offset};
  }
  static Action regPlusDataDeref(int32_t reg, int64_t offset) {
    return Action{A_REG_PLUS_DATA_DEREF, reg, offset};
  }
  static Action loadCfaOffset(int64_t offset) {
    return Action{A_LOAD_CFA_OFFSET, D_UNDEFINED, offset};
  }

  friend std::ostream& operator<<(std::ostream& out, const Action& self) {
    switch (self.kind) {
      case A_UNDEFINED:
        out << 'u';
        break;
      case A_REG_PLUS_DATA:
        out << 'r' << (int)self.reg << " + " << self.data;
        break;
      case A_REG_PLUS_DATA_DEREF:
        out << "*(r" << (int)self.reg << " + " << self.data << ')';
        break;
      case A_LOAD_CFA_OFFSET:
        out << "*(cfa + " << self.data << ')';
        break;
    }
    return out;
  }
};

} // namespace torch::unwind

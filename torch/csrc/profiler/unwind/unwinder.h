#pragma once
#include <torch/csrc/profiler/unwind/action.h>
#include <torch/csrc/profiler/unwind/unwind_error.h>
#include <cstdint>
#include <limits>

namespace torch::unwind {

// Architecture-neutral names: pc (program counter / return address),
// fp (frame pointer: x86 RBP, aarch64 x29), sp (stack pointer).
struct UnwindState {
  int64_t pc, fp, sp;
};

struct Unwinder {
  Unwinder(Action cfa, Action ret, Action fp)
      : kind_(ret.kind == A_UNDEFINED ? END : STANDARD),
        reg_(cfa.reg),
        off_(cfa.data),
        ret_off_(ret.data),
        fp_off_(
            fp.kind == A_UNDEFINED ? std::numeric_limits<int64_t>::max()
                                   : fp.data),
        deref_(cfa.kind == A_REG_PLUS_DATA_DEREF) {
    check(cfa.reg == D_STACK_PTR || cfa.reg == D_FRAME_PTR);
    check(ret.kind == A_UNDEFINED || ret.kind == A_LOAD_CFA_OFFSET);
    if (cfa.kind == A_REG_PLUS_DATA) {
      check(fp.kind == A_LOAD_CFA_OFFSET || fp.kind == A_UNDEFINED);
    } else if (cfa.kind == A_REG_PLUS_DATA_DEREF) {
      if (fp.kind == A_REG_PLUS_DATA_DEREF) {
        check(fp.reg == cfa.reg);
        fp_off_ -= cfa.data;
      } else {
        check(fp.kind == A_UNDEFINED);
      }
    } else {
      check(false);
    }
  }
  void check(bool cond) {
    if (!cond) {
      throw UnwindError("Unwinding actions do not follow supported patterns");
    }
  }
  bool terminator() const {
    return kind_ != STANDARD;
  }
  bool isUnknown() const {
    return kind_ == UNKNOWN;
  }
  // unwinder representing some pattern unsupported in
  // current implementation
  static Unwinder unknown() {
    return Unwinder();
  }
  UnwindState run(const UnwindState& cur) const {
    UnwindState r = cur;
    r.sp = (reg_ == D_STACK_PTR ? cur.sp : cur.fp) + off_;
    r.fp = fp_off_ == std::numeric_limits<int64_t>::max()
        ? cur.fp
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        : *(int64_t*)(r.sp + fp_off_);
    if (deref_) {
      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      r.sp = *(int64_t*)r.sp;
    }
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    r.pc = *(int64_t*)(r.sp + ret_off_);

    return r;
  }

 private:
  Unwinder() : kind_(UNKNOWN), reg_(0), off_(0), ret_off_(0), fp_off_(0) {}
  enum Kind { STANDARD, END, UNKNOWN } kind_;
  uint32_t reg_;
  int64_t off_;
  int64_t ret_off_;
  int64_t fp_off_;
  bool deref_{false};
};

} // namespace torch::unwind

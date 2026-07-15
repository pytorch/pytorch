#pragma once
#include <torch/csrc/profiler/unwind/action.h>
#include <torch/csrc/profiler/unwind/unwind_error.h>
#include <cstdint>
#include <limits>

namespace torch::unwind {

struct UnwindState {
  int64_t rip, rbp, rsp;
};

struct Unwinder {
  Unwinder(Action rsp, Action rip, Action rbp)
      : kind_(rip.kind == A_UNDEFINED ? END : STANDARD),
        reg_(rsp.reg),
        off_(rsp.data),
        rip_off_(rip.data),
        rbp_off_(
            rbp.kind == A_UNDEFINED ? std::numeric_limits<int64_t>::max()
                                    : rbp.data),
        deref_(rsp.kind == A_REG_PLUS_DATA_DEREF) {
    check(rsp.reg == D_RSP || rsp.reg == D_RBP);
    check(rip.kind == A_UNDEFINED || rip.kind == A_LOAD_CFA_OFFSET);
    if (rsp.kind == A_REG_PLUS_DATA) {
      check(rbp.kind == A_LOAD_CFA_OFFSET || rbp.kind == A_UNDEFINED);
    } else if (rsp.kind == A_REG_PLUS_DATA_DEREF) {
      if (rbp.kind == A_REG_PLUS_DATA_DEREF) {
        check(rbp.reg == rsp.reg);
        rbp_off_ -= rsp.data;
      } else {
        check(rbp.kind == A_UNDEFINED);
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
    r.rsp = (reg_ == D_RSP ? cur.rsp : cur.rbp) + off_;
    r.rbp = rbp_off_ == std::numeric_limits<int64_t>::max()
        ? cur.rbp
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        : *(int64_t*)(r.rsp + rbp_off_);
    if (deref_) {
      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      r.rsp = *(int64_t*)r.rsp;
    }
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    r.rip = *(int64_t*)(r.rsp + rip_off_);

    return r;
  }

 private:
  Unwinder() : kind_(UNKNOWN), reg_(0), off_(0), rip_off_(0), rbp_off_(0) {}
  enum Kind { STANDARD, END, UNKNOWN } kind_;
  uint32_t reg_;
  int64_t off_;
  int64_t rip_off_;
  int64_t rbp_off_;
  bool deref_{false};
};

} // namespace torch::unwind

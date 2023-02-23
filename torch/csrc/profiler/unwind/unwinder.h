#pragma once
#include <stdint.h>
#include <torch/csrc/profiler/unwind/action.h>
#include <torch/csrc/profiler/unwind/unwind_error.h>
#include <iostream>
#include <limits>

// register offsets in UnwindState
enum {
  US_UNDEF = -1,
  US_RIP = 0,
  US_RBP = 1,
  US_RSP = 2,
};
struct UnwindState {
  union {
    struct {
      int64_t rip, rbp, rsp;
    };
    // int64_t r[3];
  };
};

struct Unwinder {
  Unwinder(Action rsp, Action rip, Action rbp)
      : kind_(rip.kind == A_UNDEFINED ? END : STANDARD),
        reg_(reg2us(rsp.reg)),
        off_(rsp.data),
        rip_off_(rip.data),
        rbp_off_(
            rbp.kind == A_UNDEFINED ? std::numeric_limits<int64_t>::max()
                                    : rbp.data) {
    bool standard = rsp.kind == A_REG_PLUS_DATA &&
        (rip.kind == A_UNDEFINED || rip.kind == A_LOAD_CFA_OFFSET) &&
        (rbp.kind == A_LOAD_CFA_OFFSET || rbp.kind == A_UNDEFINED);
    if (!standard) {
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
    r.rsp = (reg_ == US_RSP ? cur.rsp : cur.rbp) + off_;
    r.rip = *(int64_t*)(r.rsp + rip_off_);
    r.rbp = rbp_off_ == std::numeric_limits<int64_t>::max()
        ? cur.rbp
        : *(int64_t*)(r.rsp + rbp_off_);
    return r;
  }

 private:
  Unwinder() : kind_(UNKNOWN), reg_(0), off_(0), rip_off_(0), rbp_off_(0) {}
  // map from register numbers in info to UnwindState register array
  static int32_t reg2us(int32_t reg) {
    switch (reg) {
      case D_RBP:
        return US_RBP;
      case D_RSP:
        return US_RSP;
      case D_RIP:
        return US_RIP;
      // unused register
      case US_UNDEF:
        return US_UNDEF;
      default:
        throw UnwindError("unsupported register");
    }
  }
  enum Kind { STANDARD, END, UNKNOWN } kind_;
  uint32_t reg_;
  int64_t off_;
  int64_t rip_off_;
  int64_t rbp_off_;
};

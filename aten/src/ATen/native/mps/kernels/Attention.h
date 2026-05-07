#pragma once

// Shared scalar parameters for varlen SGMMA attention kernels.
// Passed as a single constant-buffer to reduce per-dispatch binding count.
// Layout is identical in Metal (uint/int/float) and C++ (unsigned int/int/float).
struct VarlenAttnParams {
    unsigned int total_q;
    unsigned int total_k;
    float        sc;
    unsigned int gqa;
    int          wnd_left;
    int          wnd_right;
    unsigned int has_alibi;
};

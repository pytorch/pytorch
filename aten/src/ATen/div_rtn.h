#pragma once

// Integer division rounding to -Infinity
template<typename T>
static inline T div_rtn(T x, T y) {
    int q = x/y;
    int r = x%y;
    if ((r!=0) && ((r<0) != (y<0))) --q;
    return q;
}

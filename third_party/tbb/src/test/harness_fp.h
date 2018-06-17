/*
    Copyright (c) 2005-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

// include system header to prevent standard library to be included under private=public first time
#include <cstddef>
#define private public
#include "tbb/tbb_machine.h"
#undef private
#include "harness_assert.h"

#if ( __TBB_x86_32 || __TBB_x86_64 ) && __TBB_CPU_CTL_ENV_PRESENT && !defined(__TBB_WIN32_USE_CL_BUILTINS)

const int FE_TONEAREST = 0x0000,
          FE_DOWNWARD = 0x0400,
          FE_UPWARD = 0x0800,
          FE_TOWARDZERO = 0x0c00,
          FE_RND_MODE_MASK = FE_TOWARDZERO,
          SSE_RND_MODE_MASK = FE_RND_MODE_MASK << 3,
          SSE_DAZ = 0x0040,
          SSE_FTZ = 0x8000,
          SSE_MODE_MASK = SSE_DAZ | SSE_FTZ,
          SSE_STATUS_MASK = 0x3F;

const int NumSseModes = 4;
const int SseModes[NumSseModes] = { 0, SSE_DAZ, SSE_FTZ, SSE_DAZ | SSE_FTZ };

#if _WIN64 && !__TBB_X86_MSVC_INLINE_ASM_AVAILABLE && !__MINGW64__
// MinGW uses inline implementation from tbb/machine/linux_intel64.h
// and when inline asm is not available, the library uses out of line assembly which is not exported
// thus reimplementing them here

#include <float.h>

inline void __TBB_get_cpu_ctl_env ( tbb::internal::cpu_ctl_env* fe ) {
    fe->x87cw = short(_control87(0, 0) & _MCW_RC) << 2;
    fe->mxcsr = _mm_getcsr();
}
inline void __TBB_set_cpu_ctl_env ( const tbb::internal::cpu_ctl_env* fe ) {
    ASSERT( (fe->x87cw & FE_RND_MODE_MASK) == ((fe->x87cw & FE_RND_MODE_MASK) >> 2 & _MCW_RC) << 2, "Check float.h constants" );
    _control87( (fe->x87cw & FE_RND_MODE_MASK) >> 6, _MCW_RC );
    _mm_setcsr( fe->mxcsr );
}

#endif /*  _WIN64 && !__TBB_X86_MSVC_INLINE_ASM_AVAILABLE && !__MINGW64__ */

inline int GetRoundingMode ( bool checkConsistency = true ) {
    tbb::internal::cpu_ctl_env ctl;
    ctl.get_env();
    ASSERT( !checkConsistency || (ctl.mxcsr & SSE_RND_MODE_MASK) >> 3 == (ctl.x87cw & FE_RND_MODE_MASK), NULL );
    return ctl.x87cw & FE_RND_MODE_MASK;
}

inline void SetRoundingMode ( int mode ) {
    tbb::internal::cpu_ctl_env ctl;
    ctl.get_env();
    ctl.mxcsr = (ctl.mxcsr & ~SSE_RND_MODE_MASK) | (mode & FE_RND_MODE_MASK) << 3;
    ctl.x87cw = short((ctl.x87cw & ~FE_RND_MODE_MASK) | (mode & FE_RND_MODE_MASK));
    ctl.set_env();
}

inline int GetSseMode () {
    tbb::internal::cpu_ctl_env ctl;
    ctl.get_env();
    return ctl.mxcsr & SSE_MODE_MASK;
}

inline void SetSseMode ( int mode ) {
    tbb::internal::cpu_ctl_env ctl;
    ctl.get_env();
    ctl.mxcsr = (ctl.mxcsr & ~SSE_MODE_MASK) | (mode & SSE_MODE_MASK);
    ctl.set_env();
}

#elif defined(_M_ARM) || defined(__TBB_WIN32_USE_CL_BUILTINS)
const int NumSseModes = 1;
const int SseModes[NumSseModes] = { 0 };

inline int GetSseMode () { return 0; }
inline void SetSseMode ( int ) {}

const int FE_TONEAREST = _RC_NEAR,
          FE_DOWNWARD = _RC_DOWN,
          FE_UPWARD = _RC_UP,
          FE_TOWARDZERO = _RC_CHOP;

inline int GetRoundingMode ( bool = true ) {
    tbb::internal::cpu_ctl_env ctl;
    ctl.get_env();
    return ctl.my_ctl;
}
inline void SetRoundingMode ( int mode ) {
    tbb::internal::cpu_ctl_env ctl;
    ctl.my_ctl = mode;
    ctl.set_env();
}

#else /* Other archs */

#include <fenv.h>

const int RND_MODE_MASK = FE_TONEAREST | FE_DOWNWARD | FE_UPWARD | FE_TOWARDZERO;

const int NumSseModes = 1;
const int SseModes[NumSseModes] = { 0 };

inline int GetRoundingMode ( bool = true ) { return fegetround(); }
inline void SetRoundingMode ( int rnd ) { fesetround(rnd); }

inline int GetSseMode () { return 0; }
inline void SetSseMode ( int ) {}

#endif /* Other archs */

const int NumRoundingModes = 4;
const int RoundingModes[NumRoundingModes] = { FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO };
const int numFPModes = NumRoundingModes*NumSseModes;

inline void SetFPMode( int mode ) {
    SetRoundingMode( RoundingModes[mode/NumSseModes%NumRoundingModes] );
    SetSseMode( SseModes[mode%NumSseModes] );
}

#define AssertFPMode( mode ) { \
    ASSERT( GetRoundingMode() == RoundingModes[mode/NumSseModes%NumRoundingModes], "FPU control state has not been set correctly." ); \
    ASSERT( GetSseMode() == SseModes[mode%NumSseModes], "SSE control state has not been set correctly." ); \
}

inline int SetNextFPMode( int mode, int step = 1 ) {
    const int nextMode = (mode+step)%numFPModes;
    SetFPMode( nextMode );
    return nextMode;
}

class FPModeContext {
    int origSse, origRounding;
    int currentMode;
public:
    FPModeContext(int newMode) {
        origSse = GetSseMode();
        origRounding = GetRoundingMode();
        SetFPMode(currentMode = newMode);
    }
    ~FPModeContext() {
        assertFPMode();
        SetRoundingMode(origRounding);
        SetSseMode(origSse);
    }
    int setNextFPMode() {
        assertFPMode();
        return currentMode = SetNextFPMode(currentMode);
    }
    void assertFPMode() {
        AssertFPMode(currentMode);
    }
};

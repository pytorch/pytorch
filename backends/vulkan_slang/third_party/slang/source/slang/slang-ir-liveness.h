// slang-ir-liveness.h
#ifndef SLANG_IR_LIVENESS_H
#define SLANG_IR_LIVENESS_H

#include "../core/slang-list.h"
#include "slang-ir.h"

namespace Slang
{

/*

Liveness
========

## Status

Currently liveness tracking only tracks local variables in functions.

In particular it doesn't handle:

* PHI temporaries
* Variables that aren't constructed via IRVar
* Variables that might be introduced via slang-ir-restructure-scoping.h/.cpp
* Possible variable 'escape' via pointers (shouldn't be possible right now - but restructuring might
introduce issues)
* Any tracking around undefined values

If enabled output source will output with SLANG_LIVE_START and SLANG_LIVE_END macros. It's a user
space problem as to how to use these definitions (for example by adding a prelude).

## Motivation

At a first approximation liveness means variable is `in scope`. The underlying issue might be
described as

```HLSL
struct SomeStruct
{
    int value;
    int large[100];
};

int someFunction()
{
    int result = 0;

    for (int i = 0; i < ...; ++i)
    {
        doSomething();

        SomeStruct s;
        s.value = ...;

        doSomethingElse(s);

        result += s.value;
    }

    return result;
}
```

A compiler might hoist `s` outside of the loop, looking something more like...

```
int someFunction()
{
    SomeStruct s;
    int result = 0;

    for (int i = 0; i < ...; ++i)
    {
        doSomething();

        s.value = ...;

        result += doSomethingElse(s);
    }

    return result;
}
```

The problem is that now `s` is in scope over the loop, and there is potential for values from one
interation to be used in the next iteration. This isn't a problem in the original version because it
is 'obvious' that a new `s` is constructed each iteration. The key observation being that when
doSomething is executing, `s` doesn't exist, and so doesn't need to take any register space.

Why hoist? Some compilers define variables via `alloca`s, and these allocas can only be placed at
the start of the function. That being the case their scoping for where the contents is 'live' is
lost.

So liveness here is adding additional information about variables use. The start of the range is
where there is a 'fresh' copy of the variable, and the end is where where the values held in the
variable can no longer alter execution results.

## Discussion

The previous discussion of liveness could be described as being at the 'variable' level.

Liveness could be tracked in a more fine grain manner - such as tracking field liveness. `s` has no
`__init` and isn't initialized in any way. s.value does set some state, but `large` is untouched. So
in a sense s.value holds *all* of the state of s at that point, and only s.value would need to be
stored to reconstruct s (the rest could be undefined).

Is this more nuanced information useful to a downstream compilation? Maybe, but the downstream
compiler could perform all the same analysis. All it's really missing is knowing when there is a
`fresh version` of s.

How does this apply to undefined values?

```
int someFunction()
{
    int result = 0;
    int v;				/// v's value is undefined

    for (int i = 0; i < ...; ++i)
    {
        doSomething();

        SomeStruct s;
        s.value = v;

        result += doSomethingElse(s);
    }

    return result;
}
```

In this somewhat silly example, s.value is set to an undefined value. At one level you could say
that s is *all* in an undefined state, and therefore s is stateless. That's not quite right though
because although v is undefined, it should probably be the same value every loop.

Like before though, this may not matter too much in practice because a downstream compiler can see
this behavior, and handle appropriately.

Another way a compiler could `see` that it has a `fresh copy` within the loop, would be for all it's
state to be set.

```
int someFunction()
{
    SomeStruct s;

    int result = 0;

    for (int i = 0; i < ...; ++i)
    {
        doSomething();

        // (Note the syntax here is not Slang/HLSL, it's just meant to mean 'initialize s')
        s = SomeStruct{};

        s.value = v;

        result += doSomethingElse(s);
    }

    return result;
}
```

Here because of the initialization of *all* of `s`, a downstream compiler can infer that during
`doSomething` it doesn't have to potentially store the contents of `s` because it will be wiped out
after the function.

All of this gets more complex around branches. But again that is something a downstream compiler can
track if it has a way of knowing when a variable is in scope. Similarly calling into a function
could return a struct that contains fields which aren't set - this is something a downstream
compiler could determine when fully specialized.
*/

/* The mode for liveness tracking.

Currently just controls enabling/disabling, but could be used to control other aspects. */
enum class LivenessMode
{
    Disabled,
    Enabled,
};

// Helper for testing if liveness is enabled.
SLANG_FORCE_INLINE bool isEnabled(LivenessMode mode)
{
    return mode != LivenessMode::Disabled;
}

struct LivenessUtil
{
    /// Locate all of the variables across the module and add live range starts.
    static void addVariableRangeStarts(IRModule* module, LivenessMode mode);

    /// Adds LiveRangeEnd instructions to demark the end of all of the liveness starts in the module
    static void addRangeEnds(IRModule* module, LivenessMode mode);
};

} // namespace Slang

#endif // SLANG_IR_LIVENESS_H

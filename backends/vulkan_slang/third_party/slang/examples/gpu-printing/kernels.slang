// kernels.slang

// This file demonstrates how ordinary shader code can make use of
// a standalone GPU printing library implemented in `printing.slang`.
//
// The first step for using a module of Slang code is to `import` it.
//
import printing;
//
// The `import` declaration above brings all the types and functions
// declared in the `printing` module (in `printing.slang`) into
// scope so that our code can use them.

// For simplicity, we will define a single compute shader that does
// some simple printing.
//
[shader("compute")]
[numthreads(32)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    // The `printing` module defines two main printing routines.
    //
    // The first is a `println` function in the style of Java,
    // which takes zero or more arguments and prints them all
    // followed by a newline.
    //
    // HACK: We are having to explicitly call `getStringHash` here
    // because the print implementation wants to write out strings
    // in terms of their hash code, and the current Slang implementation
    // of `getStringHash` only applies to string literals.
    //
    println("hello from thread number ", tid.x);

    // The second facility supported by `printing.slang` is a C-style
    // `printf()` function.
    //
    printf_("printf from thread 0x%x\n", tid.x);
}

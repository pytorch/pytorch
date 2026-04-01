// Disbaled because Slang does semantic checks now, not the downstream compiler.

//TEST_IGNORE_FILE
//TEST(smoke):COMPARE_HLSL: -profile ps_4_0 -entry main

// We need to confirm that when there is an error in
// the input code, we allow the downstream compiler
// to detect and report the error, not us...

// A key goal here is that errors get reported at
// the right source location, ideally including
// all of file, line, and column info.

// This file used to have a parse error (missing semicolon),
// but at this point we need to parse function bodies, even
// if we don't check them, so we can't avoid reporting that one.
//
// I'm switching it to a type error instead:

struct S { int x; };

float4 main() : SV_Target
{
    float a = 1.0;

    // Invalid assignment:
    S s = a;

    float c = a + b;

    return float4(c);
}
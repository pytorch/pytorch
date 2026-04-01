// precompiled-dxil-generics.slang

// A test that uses slang-modules with embedded precompiled DXIL and a library containing generics.
// The test compiles a library slang (export-library-generics.slang) with -embed-downstream-ir then links the
// library to entrypoint slang (this file).
// The test passes if there is no errror thrown.
// TODO: Check if final linkage used only the precompiled dxil.

//TEST(windows):COMPILE: tests/library/export-library-generics.slang -o tests/library/export-library-generics.slang-module -target dxil -embed-downstream-ir -profile lib_6_6 -incomplete-library
//TEST(windows):COMPILE: tests/library/precompiled-dxil-generics.slang -target dxil -stage anyhit -entry anyhit -o tests/library/linked.dxil

import "export-library-generics";

struct Payload
{
    int val;
}

struct Attributes
{
    float2 bary;
}

[shader("anyhit")]
void anyhit(inout Payload payload, Attributes attrib)
{
    payload.val = normalFunc(floor(x * y), x) + normalFuncUsesGeneric(y);
}
